import os
import json
import yaml
import re
import time
import logging
import shutil
import random
from datetime import datetime
import cv2  
import yt_dlp
import subprocess
from video_tools import VideoTools
from audio_qa_pipeline import AudioDeepSearchPipeline
from benchmark_filter import BenchmarkFilter
from audio_necessity_tester import AudioNecessityTester
from visual_necessity_tester import VisualNecessityTester 
from async_utils import run_in_thread
import asyncio
logger = logging.getLogger("AudioTracingPipeline")

class AudioTracingPipeline(AudioDeepSearchPipeline):
    def __init__(self, config_path="config/config.yaml", prompt_path="config/prompts.yaml"):

        super().__init__(config_path, prompt_path)
        
        base_ws = self.config.get('acquisition', {}).get('workspace', 'data_workspace')
        self.trace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), base_ws, "trace"))
        self.download_dir = os.path.join(self.trace_dir, "downloads")
        self.completed_log_path = os.path.join(self.trace_dir, "completed_themes_history.json")

        self.history_lock = asyncio.Lock()
        self.file_lock = asyncio.Lock()    

        self.MAX_CONTEXT_FRAMES = 64
        self.max_workers = self.config.get('max_threads', 20)
        self.semaphore = asyncio.Semaphore(self.max_workers)
        self.vt = VideoTools()

        if not os.path.exists(self.completed_log_path):
            with open(self.completed_log_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

        self.categories_info = {
            "ENV": "Famous natural phenomena or iconic city soundscapes.",
            "SPEECH": "Iconic speeches or interviews by globally recognized figures.",
            "MUSIC": "Legendary live performances or high-fidelity instrumental solos.",
            "BIO": "Distinctive animal vocalizations or bio-acoustic events (e.g., BBC Earth)."
        }
        self.output_file = os.path.join(self.run_dir, "benchmark_audio_tracing.json")
        self.filtered_file = os.path.join(self.run_dir, "benchmark_audio_tracing_FILTERED.json")
        self.rejected_file = os.path.join(self.run_dir, "benchmark_audio_tracing_REJECTED.json")
        self.media_output_dir = os.path.join(self.run_dir, "audio_clips")
        self.image_output_dir = os.path.join(self.run_dir, "ground_truth_images")
        
        os.makedirs(self.media_output_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

        self.final_benchmark = {cat: [] for cat in self.categories_info.keys()}

    def _create_fresh_run_dir(self, suffix=""):
        base_ws = self.config.get('acquisition', {}).get('workspace', 'data_workspace')
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        local_trace_dir = os.path.join(current_script_dir, base_ws, "trace")
        
        base_runs_dir = os.path.join(local_trace_dir, "benchmark_runs")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_TRACING')
        new_dir = os.path.join(base_runs_dir, timestamp)

        os.makedirs(new_dir, exist_ok=True)
        return new_dir
            
    def sanitize_name(self, name):
        import re
        return re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))

    async def _audit_final_audio(self, category, wav_path):
        try:
            if category.upper() == "BIO":
                requirement = "Is this audio strictly BIOLOGICAL (e.g. animal cries, bird songs, insect sounds)? It must NOT be just wind, rain, or human noise."
            else:
                requirement = "Is this audio strictly ENVIRONMENTAL (e.g. thunder, heavy rain on metal, engine roar, fire crackling)? It must NOT contain prominent animal calls."
            sys_prompt = (
                "You are an Audio Content Auditor. Your task is to verify if the provided audio clip matches its intended category.\n"
                "Constraints:\n"
                "1. If the audio is ambiguous or contains too much noise/silence, reject it.\n"
                "2. **Human Speech Limit**: The total duration of human speech (including background chatter, whispers, or narration) must NOT exceed 20% of the total clip length. If speech is dominant or exceeds this 20% threshold, set 'match' to false.\n"
                "3. The primary acoustic signature must strictly belong to the specified category.\n"
                "4. **Music Policy**: Faint background music is PERMITTED, but it MUST be significantly quieter and less prominent than the thematic subject audio. If the music's volume is comparable to or louder than the theme, or if it drowns out the subject, reject it (match: false).\n"
                "Return ONLY a JSON object: {\"match\": true/false, \"reason\": \"...\", \"estimated_speech_percent\": \"...%\"}"
            )
            
            res = await self.llm.agenerate(
                system_prompt=sys_prompt,
                user_prompt=f"Category: {category}\nRequirement: {requirement}\nAnalyze the provided audio file.",
                media_files=[wav_path]
            )
            audit_res = self._parse_llm_json(res)
            
            is_match = audit_res.get('match', False)
            if not is_match:
                logger.warning(f"      [Audit Reason]: {audit_res.get('reason', 'Unknown')}")
            return is_match
        except Exception as e:
            logger.error(f"      [Audit Error] Failed to contact auditor: {e}")
            return False

    async def _get_full_video_context(self, video_id, video_file, workspace_dir):
        """
        利用 VideoTools 工具从整个视频中均匀抽取最多 MAX_CONTEXT_FRAMES 帧
        """
        def _sync_cv2_get_duration():
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur = total_frames / fps if fps > 0 else 0
            cap.release()
            return dur
            
        duration = await run_in_thread(_sync_cv2_get_duration)
        
        interval = max(0.5, duration / (self.MAX_CONTEXT_FRAMES * 1.5))
        
        temp_workspace = os.path.join(workspace_dir, f"temp_context_{video_id}_{int(time.time())}")
    
        all_slices = await self.vt.process_multimodal_sampling(
            video_manifest={video_id: {"path": video_file}},
            interval=interval,
            workspace=temp_workspace,
            need_audio=False
        )
        
        selected_slices = self.vt.uniform_select_slices(all_slices, self.MAX_CONTEXT_FRAMES)
        
        context_images = [s['image_path'] for s in selected_slices if os.path.exists(s['image_path'])]
        return context_images, temp_workspace

    async def run_acquisition_stage(self):
        logger.info("========== STAGE 0: ACQUISITION (CONCURRENT) ==========")
        acq_cfg = self.config.get('acquisition', {})
        theme_count = acq_cfg.get('themes_per_domain', 5)
        queries_per_theme = acq_cfg.get('queries_per_theme', 2)
        search_depth = acq_cfg.get('videos_per_query', 2)
        p_acq = self.prompts.get('tracing_acquisition', {})

        cat_tasks = []
        for cat_id, cat_desc in self.categories_info.items():
            cat_tasks.append(self._process_category_acquisition(
                cat_id, cat_desc, theme_count, queries_per_theme, search_depth, p_acq, acq_cfg
            ))
        
        await asyncio.gather(*cat_tasks)

    async def _process_category_acquisition(self, cat_id, cat_desc, theme_count, queries_per_theme, search_depth, p_acq, acq_cfg):
        async with self.history_lock:
            try:
                with open(self.completed_log_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                category_historical_themes = history_data.get(cat_id, [])
            except:
                category_historical_themes = []

        exclusion_str = ", ".join(category_historical_themes) if category_historical_themes else "None"
        
        logger.info(f"[*] Brainstorming {theme_count} NEW themes for {cat_id}. (Excluding {len(category_historical_themes)} existing {cat_id} items)")

        diversity_instruction = (
            f"\n\n[STRICT CATEGORY CONSTRAINT: {cat_id}]\n"
            f"You are currently generating themes specifically for the {cat_id} category ({cat_desc}).\n"
            f"The following {cat_id} themes already exist in our database and MUST BE EXCLUDED:\n"
            f"--- ALREADY COMPLETED {cat_id} THEMES ---\n"
            f"{exclusion_str}\n"
            f"------------------------------------------\n"
            f"Please provide {theme_count} NEW, HIGH-DIVERSITY themes that do not overlap with the list above."
        )

        if cat_id == "ENV":
            user_p = p_acq['env_masterpiece_brainstormer']['user_template'].format(count=theme_count) + diversity_instruction
            sys_p = p_acq['env_masterpiece_brainstormer']['system']
        elif cat_id == "BIO":
            user_p = p_acq['bio_masterpiece_brainstormer']['user_template'].format(count=theme_count) + diversity_instruction
            sys_p = p_acq['bio_masterpiece_brainstormer']['system']
        else:
            user_p = p_acq['brainstormer']['user_template'].format(count=theme_count, cat_id=cat_id, cat_desc=cat_desc) + diversity_instruction
            sys_p = p_acq['brainstormer']['system']

        res = await self.llm.agenerate(sys_p, user_p)
        themes = self._parse_llm_json(res)
        if not isinstance(themes, list): return

        theme_tasks = []
        for theme in themes:
            theme_tasks.append(self._process_theme_acquisition(
                cat_id, theme, queries_per_theme, search_depth, p_acq, acq_cfg
            ))
        await asyncio.gather(*theme_tasks)

    async def _process_theme_acquisition(self, cat_id, theme, queries_per_theme, search_depth, p_acq, acq_cfg):
        folder_name = self.sanitize_name(theme[:40])
        save_path = os.path.join(self.download_dir, cat_id, folder_name)
        
        if os.path.exists(save_path): 
            logger.info(f"    [SKIP] Theme folder exists: {folder_name}")
            return
        if cat_id == "ENV":
            q_res = await self.llm.agenerate(
                p_acq['env_masterpiece_brainstormer']['query_generator']['system'],
                p_acq['env_masterpiece_brainstormer']['query_generator']['user_template'].format(count=queries_per_theme, theme=theme)
            )
        else:
            q_res = await self.llm.agenerate(
                p_acq['query_generator']['system'],
                p_acq['query_generator']['user_template'].format(count=queries_per_theme, theme=theme)
            )

        queries = self._parse_llm_json(q_res)
        if not queries: return

        query_tasks = []
        for query in queries:
            query_tasks.append(self._process_query_task(
                cat_id, theme, query, save_path, search_depth, acq_cfg
            ))
        await asyncio.gather(*query_tasks)

    async def _process_query_task(self, cat_id, theme, query, save_path, search_depth, acq_cfg):
        async with self.semaphore:
            logger.info(f"    Searching: {query}")
            results = await self.vt.search_videos(query, self.config, max_results=search_depth)
            if not results: return

            min_views = acq_cfg.get('min_view_count', 100000)
            min_subs = acq_cfg.get('min_subscriber_count', 100000)
            MAX_DUR, MIN_DUR = 1800, 30

            for v in results:
                stats = await self.vt._get_channel_stats(v['url'])
                
                v_views = stats.get('views', 0)
                v_subs = stats.get('subs', 0)
                v_dur = stats.get('duration', 0)

                if not (MIN_DUR <= v_dur <= MAX_DUR): continue
                if v_views < min_views or v_subs < min_subs: continue

                logger.info(f"      [PASS] Found Video: {v['title']} (Views: {v_views})")

                manifest = await self.vt.download_videos([v['url']], save_path, self.config, height=720)
                if manifest:
                    vid_id = list(manifest.keys())[0]
                    meta = await self.vt.fetch_video_metadata(v['url'], self.config)
                    
                    await self._perform_slicing(vid_id, manifest[vid_id]['path'], save_path)
                    
                    await self._update_theme_meta(save_path, theme, cat_id, vid_id, v, meta)
                    
                    logger.info(f"    [SUCCESS] Captured: {v['title']}")
                    break 

    async def _update_theme_meta(self, save_path, theme, cat_id, vid_id, v, meta):
        """提取出的写文件逻辑，确保异步安全"""
        meta_file = os.path.join(save_path, "theme_meta.json")

        async with self.file_lock:
            duration_sec = meta.get('duration', 0)
            dur_str = f"{duration_sec // 60:02d}:{duration_sec % 60:02d}"
            
            theme_data = {"theme": theme, "domain": cat_id, "videos": []}
            
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        theme_data = json.load(f)
                except: pass

            new_video_entry = {
                "id": vid_id, 
                "title": meta.get('full_title', v['title']), 
                "channel": meta.get('uploader', 'Unknown'),
                "upload_date": meta.get('upload_date', 'Unknown'),
                "duration_str": dur_str,
                "url": v['url'], 
                "description": meta.get('full_description', '')
            }

            if vid_id not in [vid['id'] for vid in theme_data.get('videos', [])]:
                theme_data['videos'].append(new_video_entry)
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(theme_data, f, indent=2, ensure_ascii=False)

    async def run_pipeline(self):
        logger.info("========== STAGE 1: AUDIO TRACING GENERATION ==========")
        abs_download_dir = os.path.abspath(self.download_dir)
        target_categories = list(self.categories_info.keys()) 

        for cat in target_categories:
            cat_path = os.path.join(abs_download_dir, cat)
            if not os.path.exists(cat_path):
                continue
            
            themes = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
            for theme in themes:
                theme_path = os.path.join(cat_path, theme)
                meta_path = os.path.join(theme_path, "theme_meta.json")
                if not os.path.exists(meta_path): continue

                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                for v_info in meta.get('videos', []):
                    await self._process_tracing_video(cat, meta['theme'], v_info, theme_path)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.final_benchmark, f, indent=2, ensure_ascii=False)

    async def _process_tracing_video(self, category, theme, video_info, theme_path):
        video_id = video_info['id']
        video_file = self._find_video_file(theme_path, video_id)
        if not video_file: return

        sample_folder = os.path.join(theme_path, f"samples_{video_id}")
        timed_files = self._get_sorted_timed_files(sample_folder)
        if not timed_files: return

        if category.upper() in ["ENV", "BIO"]:
            await self._run_env_bio_batch_scan(category, video_info, video_file, sample_folder, timed_files)
        else:
            await self._process_with_original_logic(category, video_info, video_file, sample_folder, timed_files)
    def _get_existing_themes_for_cat(self, cat_id):
        cat_path = os.path.join(self.download_dir, cat_id)
        if not os.path.exists(cat_path):
            return []
        return [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
    
    async def _run_env_bio_batch_scan(self, category, video_info, video_file, sample_folder, timed_files):
        request_count = 0
        max_requests = 10
        video_id = video_info['id']
        for minute in range(0, 60): 
            if request_count >= max_requests:
                logger.info(f"      [INFO] Max requests ({max_requests}) reached for this video. Stopping scan.")
                break
                
            target_start_sec = minute * 60
            matching_indices = [i for i, x in enumerate(timed_files) if x['sec'] >= target_start_sec]
            if not matching_indices:
                break 
            
            idx = matching_indices[0]
            batch_items = timed_files[idx : idx + 2]
            if len(batch_items) < 2:
                break 
            
            wav_paths = [os.path.abspath(os.path.join(sample_folder, x['file'])) for x in batch_items]
            current_ts_label = batch_items[0]['ts_str']
            
            try:
                request_count += 1
                
                if category.upper() == "BIO":
                    type_constraint = "STRICT REQUIREMENT: This is a BIO (Biological) clip"
                else:
                    type_constraint = "STRICT REQUIREMENT: This is an ENV (Environmental) clip"

                purity_prompt = self.prompts.get('acoustic_purity_scanner', {})
                res = await self.llm.agenerate(
                    system_prompt=purity_prompt['system'],
                    user_prompt=f"{type_constraint}\n\nAnalyze this 20s segment. Title: {video_info['title']}",
                    media_files=wav_paths
                )
                scan_res = self._parse_llm_json(res)
                if not scan_res or not scan_res.get('is_pure_acoustic'):
                    continue 
                
                logger.info(f"      [MATCH] Found potential {category} at {current_ts_label}!")

                subject = scan_res.get('acoustic_description', f"Pure {category} acoustic sequence")
                s_off = float(scan_res.get('safe_start_offset', 0))
                e_off = float(scan_res.get('safe_end_offset', 20))
                
                if (e_off - s_off) < 5.0:
                    logger.warning(f"      [SKIP] Segment too short: {e_off - s_off}s (requires 5s+)")
                    continue 

                context_images, temp_ws = await self._get_full_video_context(video_id, video_file, os.path.dirname(video_file))
                
                if not context_images:
                    logger.warning(f"      [SKIP] No visual context frames could be extracted.")
                    shutil.rmtree(temp_ws, ignore_errors=True)
                    continue

                try:
                    sample_base_name = f"TRACING_{category}_{video_info['id']}_{current_ts_label}"
                    final_wav_path = os.path.join(self.media_output_dir, f"{sample_base_name}.wav")
                    
                    logger.info(f"      [INFO] Attempting to crop/merge audio to: {final_wav_path}")
                    if await self._crop_and_merge_audio(wav_paths, s_off, e_off, final_wav_path):
                        
                        logger.info(f"      [AUDIT] Starting secondary quality audit...")
                        if not await self._audit_final_audio(category, final_wav_path):
                            logger.warning(f"      [SKIP] Failed secondary audit (category mismatch or too much speech).")
                            if os.path.exists(final_wav_path): os.remove(final_wav_path)
                            continue
                        
                        target_img_dir = os.path.join(self.image_output_dir, sample_base_name)
                        os.makedirs(target_img_dir, exist_ok=True)
                        for img_path in context_images:
                            shutil.copy2(img_path, os.path.join(target_img_dir, os.path.basename(img_path)))
                        
                        upload_year = video_info.get('upload_date', 'Unknown')[:4]
                        
                        count_before = len(self.final_benchmark[category])
                        
                        logger.info(f"      [GENERATE] Calling _generate_tracing_qa_v3...")
                        await self._generate_tracing_qa_v3(category, video_info, final_wav_path, context_images, video_info['channel'], upload_year, subject)
                        
                        if len(self.final_benchmark[category]) > count_before:
                            logger.info(f"      [SUCCESS] Item successfully added to benchmark. Task done for this video.")
                            return 
                        else:
                            logger.warning(f"      [RETRY] QA was generated but rejected (leakage or quality). Trying next minute...")
                    else:
                        logger.error(f"      [ERROR] _crop_and_merge_audio returned False. Check ffmpeg/paths.")
                finally:
                    shutil.rmtree(temp_ws, ignore_errors=True)

            except Exception as e:
                logger.error(f"      [CRITICAL ERROR] Exception during scan at minute {minute}: {str(e)}", exc_info=True)
                continue

    async def _process_with_original_logic(self, category, video_info, video_file, sample_folder, timed_files):
        """增加日志的 SPEECH 处理逻辑"""
        pick_n = min(self.num_consecutive, len(timed_files))
        video_id = video_info['id'] 
        
        logger.info(f"[SPEECH-TRACE] Starting processing for Video: {video_id}")

        for attempt in range(3):
            start_idx = random.randint(0, len(timed_files) - pick_n)
            selected_items = timed_files[start_idx: start_idx + pick_n]
            wav_paths = [os.path.abspath(os.path.join(sample_folder, x['file'])) for x in selected_items]

            logger.info(f"  - Attempt {attempt+1}: Slices from {selected_items[0]['ts_str']}")

            try:
                cat_key = category.lower()
                sys_p = self.prompts.get(cat_key, {}).get('investigator_system', "")
                sys_p = sys_p.replace("{num_slices}", str(pick_n)).replace("{slice_duration}", "10.0").replace("{total_duration}", str(pick_n*10.0))

                res = await self.llm.agenerate(sys_p, f"Title: {video_info['title']}", media_files=wav_paths)
                seg_analysis = self._parse_llm_json(res)
                
                if not seg_analysis or not seg_analysis.get('segment_analysis', {}).get('is_safe'):
                    logger.warning(f"  - LLM marked segment as unsafe or invalid. Skipping.")
                    continue
                
                analysis = seg_analysis['segment_analysis']
                s_off = analysis['safe_start_offset_seconds']
                e_off = analysis['safe_end_offset_seconds']
                
                logger.info(f"  - LLM suggested offsets: {s_off}s to {e_off}s")

                base_ts = selected_items[0]['ts_str']
                sample_base_name = f"TRACING_{category}_{video_id}_{base_ts}"
                
                final_wav_path = os.path.join(self.media_output_dir, f"{sample_base_name}.wav")
                logger.info(f"  - Final target: {final_wav_path}")

                os.makedirs(self.media_output_dir, exist_ok=True)

                if await self._crop_and_merge_audio(wav_paths, s_off, e_off, final_wav_path):
                    target_img_dir = os.path.join(self.image_output_dir, sample_base_name)
                    os.makedirs(target_img_dir, exist_ok=True)
                    
                    context_images, temp_ws = await self._get_full_video_context(video_id, video_file, os.path.dirname(video_file))
                    if context_images:
                        for img_path in context_images:
                            shutil.copy2(img_path, os.path.join(target_img_dir, os.path.basename(img_path)))
                        logger.info(f"  - Visual context saved to: {target_img_dir}")
                    
                        author_name = video_info.get('channel', 'Unknown')
                        upload_year = video_info.get('upload_date', 'Unknown')[:4]
                        subject = seg_analysis.get('identity_grounding', {}).get('full_identity', "Unknown Speaker")

                        await self._generate_tracing_qa_v3(category, video_info, final_wav_path, context_images, author_name, upload_year, subject)
                        shutil.rmtree(temp_ws, ignore_errors=True)
                        break 
                    else:
                        logger.error(f"  - Failed to extract visual context frames.")
                else:
                    logger.error(f"  - _crop_and_merge_audio returned False for {sample_base_name}")
            except Exception as e:
                logger.error(f"  - Unexpected error in Speech Logic: {e}", exc_info=True)
                continue

    def _get_sorted_timed_files(self, sample_folder):
        if not os.path.exists(sample_folder): return []
        all_files = [f for f in os.listdir(sample_folder) if f.endswith('.wav')]
        timed = []
        for f in all_files:
            match = re.search(r'_(?P<ts>\d+m\d+s)\.wav$', f)
            if match:
                ts_str = match.group('ts')
                timed.append({
                    "sec": self._parse_timestamp_to_seconds(ts_str),
                    "ts_str": ts_str,
                    "file": f
                })
        return sorted(timed, key=lambda x: x['sec'])

    def _find_video_file(self, theme_path, video_id):
        for f in os.listdir(theme_path):
            if f.startswith(video_id) and f.endswith(('.mp4', '.mkv', '.webm')):
                return os.path.abspath(os.path.join(theme_path, f))
        return None
    def _get_segment_video_context(self, video_id, video_file, start_sec, end_sec, workspace_dir):
        temp_workspace = os.path.join(workspace_dir, f"temp_seg_{video_id}_{int(time.time())}")
        os.makedirs(temp_workspace, exist_ok=True)
        
        context_images = []
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            logger.error(f"      [V-ERROR] Could not open video file: {video_file}")
            return [], temp_workspace

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25 
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        start_sec = max(0, start_sec)
        end_sec = min(duration, end_sec)

        num_frames = 16
        if end_sec - start_sec < 0.5: 
            sample_times = [start_sec] * num_frames
        else:
            sample_times = [start_sec + (end_sec - start_sec) * i / (num_frames - 1) for i in range(num_frames)]
        
        logger.info(f"      [V-DEBUG] Sampling 16 frames between {start_sec:.2f}s and {end_sec:.2f}s")

        for i, t in enumerate(sample_times):
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                img_name = f"context_{i:02d}.jpg"
                img_path = os.path.join(temp_workspace, img_name)
                cv2.imwrite(img_path, frame)
                context_images.append(img_path)
        
        cap.release()
        
        if not context_images:
            logger.error(f"      [V-ERROR] Failed to extract any frames from {video_file}")
            
        return context_images, temp_workspace

    async def _perform_slicing(self, vid_id, video_path, theme_dir):
        unique_temp_workdir = os.path.join(theme_dir, f"workdir_{vid_id}")
        os.makedirs(unique_temp_workdir, exist_ok=True)

        await self.vt.process_multimodal_sampling(
            video_manifest={vid_id: {"path": video_path}},
            interval=self.config.get('sampling', {}).get('interval', 10.0),
            workspace=unique_temp_workdir,
            need_audio=True,
            audio_duration=self.config.get('sampling', {}).get('audio_duration', 10.0)
        )

        src_samples = os.path.join(unique_temp_workdir, "samples", vid_id)
        dst_samples = os.path.join(theme_dir, f"samples_{vid_id}")

        if os.path.exists(src_samples):
            if os.path.exists(dst_samples): 
                await run_in_thread(shutil.rmtree, dst_samples)
            await run_in_thread(os.rename, src_samples, dst_samples)

        await run_in_thread(shutil.rmtree, unique_temp_workdir, ignore_errors=True)

    async def _generate_tracing_qa_v3(self, category, video_info, wav_path, storyboard, author_name, upload_year, acoustic_identity):
        try:
            forbidden_metadata = (
                f"Title: {video_info.get('title')}\n"
                f"Description: {video_info.get('description', '')[:300]}"
            )
            user_template = self.prompts.get('audio_tracing', {}).get('user_template', "")
            if not user_template:
                logger.error("YAML user_template is missing!")
                return
            user_msg = user_template.format(
                title=video_info.get('title'),
                source_channel=author_name,
                full_identity=acoustic_identity,
                description_snippet=video_info.get('description', '')[:200],
                img_count=len(storyboard)
            )
            raw_res = await self.llm.agenerate(
                system_prompt=self.prompts.get('audio_tracing', {}).get('system'), 
                user_prompt=user_msg, 
                media_files=[wav_path] + storyboard 
            )
            
            parsed = self._parse_llm_json(raw_res)

            if not parsed or 'challenges' not in parsed: 
                logger.warning(f"      [Fail] LLM response invalid or missing 'challenges'.")
                return

            inner_challenge = parsed['challenges'][0]
            question_text = inner_challenge.get('question', '')
            
            if author_name.lower() in question_text.lower():
                logger.error(f"      [LEAKAGE DETECTED] Model leaked real author name. Discarding.")
                return

            formatted_item = {
                "sample_id": os.path.basename(wav_path).replace(".wav", ""),
                "audio_file": os.path.join("audio_clips", os.path.basename(wav_path)),
                "visual_ground_truth_dir": os.path.join("ground_truth_images", os.path.basename(wav_path).replace(".wav", "")),
                "video_info": {
                    "id": video_info['id'],
                    "title": video_info.get('title'),
                    "url": video_info.get('url'),
                    "channel": author_name,
                    "year": upload_year,
                    "identity": acoustic_identity 
                },
                "challenge": {  
                    "challenges": [
                        {
                            "question": inner_challenge.get('question'),
                            "reasoning_checklist": inner_challenge.get('reasoning_checklist', []),
                            "ground_truth_answer": inner_challenge.get('ground_truth_answer')
                        }
                    ]
                }
            }

            self.final_benchmark[category].append(formatted_item)
            logger.info(f"      [OK] Valid Intra-video multi-hop challenge created (Identity Masked).")

        except Exception as e:
            logger.error(f"      [Error] QA Generation failed: {e}")

    async def run_full_pipeline(self):
        logger.info("========== STEP 0: ACQUISITION ==========")
        await self.run_acquisition_stage() 

        logger.info("========== STEP 1: GENERATION ==========")
        await self.run_generation_stage()

        logger.info("========== STEP 2: QUALITY FILTERING (LLM Audit) ==========")
        self.output_file = os.path.join(self.run_dir, "benchmark_audio_tracing.json")
        if os.path.exists(self.output_file):
            filter_engine = BenchmarkFilter()
            await filter_engine.process_single_file(
                input_file=self.output_file, 
                output_kept_file=self.filtered_file, 
                output_rejected_file=self.rejected_file
            )
        else:
            logger.error("Raw benchmark file not found, skipping filter.")
            return
        if os.path.exists(self.filtered_file):
            necessity_tester = AudioNecessityTester()
            await necessity_tester.process_json_file(self.filtered_file)  
        else:
            logger.warning("Filtered file not found, skipping audio necessity test.")

        logger.info("========== STEP 4: VISUAL NECESSITY TESTING (Text Leakage Check) ==========")
        target_file_for_visual_test = self.filtered_file
        
        if os.path.exists(target_file_for_visual_test):
            visual_tester = VisualNecessityTester(self.config, self.prompts)
            visual_tester.process_json_file(target_file_for_visual_test)
        else:
            logger.warning("File missing, skipping Visual Necessity Test.")

        logger.info(f"========== ALL TRACING TASKS COMPLETED ==========")
        logger.info(f"Final Hard Set is in: {self.run_dir}")
    async def _process_tracing_video_safely(self, category, theme, video_info, theme_path):
        async with self.semaphore:
            try:
                await self._process_tracing_video(category, theme, video_info, theme_path)
            except Exception as e:
                logger.error(f"      [Task Error] Unhandled exception in video {video_info.get('id')}: {e}")
    async def run_generation_stage(self):
        abs_download_dir = os.path.abspath(self.download_dir)
        target_categories = list(self.categories_info.keys()) 
        
        tasks = []
        processed_themes = [] 
        for cat in target_categories:
            cat_path = os.path.join(abs_download_dir, cat)
            if not os.path.exists(cat_path): continue
            
            themes = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
            for theme in themes:
                theme_path = os.path.join(cat_path, theme)
                meta_path = os.path.join(theme_path, "theme_meta.json")
                if not os.path.exists(meta_path): continue

                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                for v_info in meta.get('videos', []):
                    tasks.append(
                        self._process_tracing_video_safely(cat, meta['theme'], v_info, theme_path)
                    )

        if tasks:
            logger.info(f"Submitting {len(tasks)} videos for concurrent tracing generation (Max Workers: {self.max_workers})...")
            await asyncio.gather(*tasks)

        await self._finalize_and_cleanup(processed_themes)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.final_benchmark, f, indent=2, ensure_ascii=False)
        logger.info(f"      [OK] Raw generation complete: {len(self.final_benchmark)} categories processed.")

    async def _finalize_and_cleanup(self, processed_themes):
        logger.info("========== POST-PROCESS: LOGGING & CLEANUP ==========")
        
        async with self.history_lock:
            with open(self.completed_log_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)

            for cat, theme_name, theme_path in processed_themes:
                if cat not in history_data:
                    history_data[cat] = []
                if theme_name not in history_data[cat]:
                    history_data[cat].append(theme_name)
                
                if os.path.exists(theme_path):
                    logger.info(f"      [Cleanup] Removing processed theme folder: {theme_path}")
                    await run_in_thread(shutil.rmtree, theme_path, ignore_errors=True)

            with open(self.completed_log_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully logged {len(processed_themes)} themes and cleaned up storage.")

if __name__ == "__main__":
    pipeline = AudioTracingPipeline()
    # pipeline.run_acquisition_stage()
    # pipeline.run_pipeline()


    asyncio.run(pipeline.run_full_pipeline())
    

    






