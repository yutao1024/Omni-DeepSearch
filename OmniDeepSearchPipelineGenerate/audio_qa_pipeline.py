import os
import json
import yaml
import re
import random
import wave
import logging
import asyncio
import aiohttp
from datetime import datetime
from llm_provider import get_llm_provider
from live_wiki_walker import LiveWikiWalker
from benchmark_filter import BenchmarkFilter
from benchmark_refiner import BenchmarkRefiner
from audio_necessity_tester import AudioNecessityTester
import requests
from requests.adapters import HTTPAdapter                    
from tqdm.asyncio import tqdm                                       
import subprocess
import asyncio
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("AudioDeepSearchPipeline")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class AudioDeepSearchPipeline:
    def __init__(self, config_path="config/config.yaml", prompt_path="config/prompts.yaml",is_interaction=False):
        config_path = os.path.join(BASE_DIR, config_path)
        prompt_path = os.path.join(BASE_DIR, prompt_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        self.llm = get_llm_provider(self.config['llm'], self.config['llm']['default_provider'])
        self.download_dir = os.path.join(BASE_DIR, self.config.get('download_dir', 'downloads'))

        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.headers.update()

        suffix = "_INTERACTION" if is_interaction else ""
        self.run_dir = self._create_fresh_run_dir(suffix=suffix)
        
        self.output_file = os.path.join(self.run_dir, "benchmark.json")
        self.media_output_dir = os.path.join(self.run_dir, "audio_clips")
        os.makedirs(self.media_output_dir, exist_ok=True)

        self.final_hard_file = os.path.join(self.run_dir, "benchmark_FINAL_HARD.json")
        self.still_easy_file = os.path.join(self.run_dir, "benchmark_STILL_EASY.json")

        self.num_consecutive = self.config.get('sampling', {}).get('max_samples_per_video', 3)
        self.slice_duration = self.config.get('sampling', {}).get('audio_duration', 10.0)
        self.half_duration = self.slice_duration / 2
        self.final_benchmark = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}

        self.interaction_dir = os.path.join(BASE_DIR, self.config.get('interaction_dir', 'data_workspace/interactions'))
        
        self.kg_walker = LiveWikiWalker(llm_client=self.llm, llm_model="Not_Used", config_path=prompt_path)

        self.max_workers = self.config.get('max_threads', 10) 
        self.semaphore = asyncio.Semaphore(self.max_workers)

    def _create_fresh_run_dir(self, suffix=""):
        base_runs_dir = os.path.join(BASE_DIR, "data_workspace", "benchmark_runs")
        os.makedirs(base_runs_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        new_dir = os.path.join(base_runs_dir, f"{timestamp}{suffix}")
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    async def _crop_and_merge_audio(self, wav_paths, start_sec, end_sec, output_path):
        logger.info(f"[AUDIO-DEBUG] Starting crop_and_merge...")
        logger.info(f"  - Source Slices: {len(wav_paths)} files")
        logger.info(f"  - Request Range: {start_sec}s to {end_sec}s")
        logger.info(f"  - Target Path: {output_path}")

        try:
            start = float(start_sec)
            end = float(end_sec)
            duration = end - start
            if duration <= 0:
                logger.error(f"[AUDIO-ERROR] Invalid duration: {duration}s (Start: {start}, End: {end})")
                return False
        except Exception as e:
            logger.error(f"[AUDIO-ERROR] Offset conversion failed: {e}")
            return False

        filter_complex = "".join([f"[{i}:a]" for i in range(len(wav_paths))])
        filter_complex += f"concat=n={len(wav_paths)}:v=0:a=1[aout]"
        
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for p in wav_paths:
            cmd.extend(["-i", p])
        
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[aout]",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}", 
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            output_path
        ])

        try:
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.wait()
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    logger.info(f"[AUDIO-SUCCESS] File created: {output_path} (Size: {file_size} bytes)")
                    return True
                else:
                    logger.error(f"[AUDIO-FAIL] File exists but is 0 bytes: {output_path}")
            else:
                logger.error(f"[AUDIO-FAIL] FFmpeg finished but file not found: {output_path}")
        except Exception as e:
            logger.error(f"[AUDIO-ERROR] Process execution failed: {e}")
        
        return False

    def _parse_timestamp_to_seconds(self, ts_str):
        match = re.match(r'(\d+)m(\d+)s', ts_str)
        if match: return int(match.group(1)) * 60 + int(match.group(2))
        return 0

    def _seconds_to_hms(self, total_seconds):
        ts = max(0, int(total_seconds))
        return f"{ts // 3600:02d}:{(ts % 3600) // 60:02d}:{ts % 60:02d}"

    def _parse_llm_json(self, text):
        try: return json.loads(text)
        except:
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
                if json_match: return json.loads(json_match.group(1))
                start, end = text.find('{'), text.rfind('}')
                if start != -1 and end != -1: return json.loads(text[start:end+1])
            except: pass
        return None

    async def _audit_audio_content(self, category, wav_path):
        requirements = {
            "BIO": "The audio MUST be strictly BIOLOGICAL (e.g., animal vocalizations, bird songs, insect noises). Reject if it's just wind, rain, or machinery.",
            "ENV": "The audio MUST be strictly ENVIRONMENTAL (e.g., weather, industrial machines, city hums, fire). Reject if animal calls are the primary sound.",
            "MUSIC": "The audio MUST be a pure instrumental or vocal musical performance. Reject if there is significant non-musical noise or talking.",
            "SPEECH": "The audio MUST be a clear recording of a person speaking. Reject if background noise or music drowns out the voice."
        }
        
        req = requirements.get(category.upper(), "Verify if the audio matches the intended context.")

        sys_prompt = (
            "You are a Senior Audio Content Auditor.\n"
            "Your task is to verify if the provided raw audio clip strictly matches the assigned category.\n"
            "Strict Criteria:\n"
            "1. The total duration of human speech (including background chatter, whispers, or narration) must NOT exceed 20% of the total clip length.(unless category is SPEECH).\n"
            "2. Correct Texture: The sound must clearly represent the category described.\n"
            "3. **Music Policy**: Faint background music is PERMITTED, but it MUST be significantly quieter and less prominent than the thematic subject audio. (unless category is MUSIC). If the music's volume is comparable to or louder than the theme, or if it drowns out the subject, reject it (match: false).\n"
            "Return ONLY a JSON object: {\"is_valid\": true/false, \"reason\": \"...\"}"
        )

        user_prompt = f"Category: {category}\nRequirement: {req}\nAnalyze the provided audio."

        try:
            raw = await self.llm.agenerate(system_prompt=sys_prompt, user_prompt=user_prompt, media_files=[wav_path])
            res = self._parse_llm_json(raw)
            if res and res.get('is_valid') is True: return True
            logger.warning(f"      [Audit Failed] Category: {category}. Reason: {res.get('reason') if res else 'Unknown'}")
            return False
        except Exception as e:
            logger.error(f"Audit Exception: {e}")
            return True

    async def run(self):
        active_categories = [cat for cat in self.final_benchmark.keys() if cat != "INTERACTION"]
        tasks_data = []

        for category in active_categories:
            category_path = os.path.join(self.download_dir, category)
            if not os.path.exists(category_path): continue
            themes = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]

            for theme in themes:
                theme_path = os.path.join(category_path, theme)
                meta_path = os.path.join(theme_path, "theme_meta.json")
                if not os.path.exists(meta_path): continue

                with open(meta_path, 'r', encoding='utf-8') as f:
                    theme_meta = json.load(f)

                for video_info in theme_meta.get('videos', []):
                    tasks_data.append((category, theme_meta.get('theme', theme), video_info, theme_path))

        logger.info(f"Submitting {len(tasks_data)} videos to Async Batch Handler with semaphore={self.max_workers}...")

        tasks = [self._process_video_consecutive(cat, theme, info, path) for cat, theme, info, path in tasks_data]
        await tqdm.gather(*tasks, desc="Processing Videos Asynchronously")

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.final_benchmark, f, indent=2, ensure_ascii=False)

    async def _process_video_consecutive(self, category, theme_display_name, video_info, theme_path):
        """处理单条视频切片的异步流水线"""
        async with self.semaphore:
            video_id = video_info['id']
            sample_folder_path = os.path.join(theme_path, f"samples_{video_id}")
            if not os.path.exists(sample_folder_path): return

            all_files = [f for f in os.listdir(sample_folder_path) if f.endswith('.wav')]
            timed_files = sorted(
                [{"sec": self._parse_timestamp_to_seconds(re.search(r'_(?P<ts>\d+m\d+s)\.wav$', f).group('ts')),
                  "ts_str": re.search(r'_(?P<ts>\d+m\d+s)\.wav$', f).group('ts'), "file": f}
                 for f in all_files if re.search(r'_(?P<ts>\d+m\d+s)\.wav$', f)], key=lambda x: x['sec'])
            if not timed_files: return

            pick_n = min(self.num_consecutive, len(timed_files))
            
            max_start_idx = max(1, len(timed_files) - pick_n + 1)
            available_indices = list(range(0, max_start_idx, pick_n)) 
            random.shuffle(available_indices) 
            
            is_success = False
            final_identity = None
            final_seg_analysis = None
            final_abs_start = 0
            final_abs_end = 0
            final_safe_start = 0
            final_safe_end = 0
            final_path_str = ""
            final_filename_str = ""

            for attempt in range(min(3, len(available_indices))):
                start_idx = available_indices[attempt]
                selected_items = timed_files[start_idx: start_idx + pick_n]
                wav_paths = [os.path.abspath(os.path.join(sample_folder_path, x['file'])) for x in selected_items]

                try:
                    identity, seg_analysis = await self._step1_investigate(category, theme_display_name, video_info, wav_paths)
                    if not seg_analysis.get('is_safe'):
                        continue 
                    safe_start = float(seg_analysis.get('safe_start_offset_seconds', 0))
                    safe_end = float(seg_analysis.get('safe_end_offset_seconds', pick_n * self.slice_duration))
                    abs_start = selected_items[0]['sec'] - self.half_duration + safe_start
                    abs_end = abs_start + (safe_end - safe_start)

                    start_time_str = self._seconds_to_hms(abs_start).replace(':', '')
                    final_filename = f"{category}_{video_id}_{start_time_str}.wav"
                    final_path = os.path.join(self.media_output_dir, final_filename)

                    if not await self._crop_and_merge_audio(wav_paths, safe_start, safe_end, final_path):
                        continue 

                    if not await self._audit_audio_content(category, final_path):
                        if os.path.exists(final_path): os.remove(final_path)
                        continue 
                    is_success = True
                    final_identity = identity
                    final_seg_analysis = seg_analysis
                    final_abs_start = abs_start
                    final_abs_end = abs_end
                    final_safe_start = safe_start
                    final_safe_end = safe_end
                    final_path_str = final_path
                    final_filename_str = final_filename
                    break

                except Exception as e:
                    logger.warning(f"    [Attempt {attempt+1}/3 Error] {e}")
                    continue
            
            if not is_success:
                return

            try:
                kg_path = await self._step2_kg_walk(final_identity, category)
                if not kg_path: return

                timing_info = {
                    "abs_start": self._seconds_to_hms(final_abs_start),
                    "abs_end": self._seconds_to_hms(final_abs_end),
                    "duration": f"{final_safe_end - final_safe_start:.1f}s",
                    "audio_file": os.path.join("audio_clips", final_filename_str)
                }

                await self._step3_generate_deep_qa(category, theme_display_name, video_info, timing_info, [final_path_str],
                                             kg_path, final_identity, final_seg_analysis)

            except Exception as e:
                logger.error(f"    [Error] Failed video {video_id}: {e}")

    async def _step1_investigate(self, category, theme, video_info, media_files):
        prompt_config = self.prompts.get(category.lower(), self.prompts.get('speech', {}))
        sys_prompt = prompt_config['investigator_system']
        total_dur = self.num_consecutive * self.slice_duration

        sys_prompt = sys_prompt.replace("{num_slices}", str(self.num_consecutive))
        sys_prompt = sys_prompt.replace("{slice_duration}", str(self.slice_duration))
        sys_prompt = sys_prompt.replace("{total_duration}", str(total_dur))

        user_prompt = f"Known Context:\n- Title: {video_info.get('title')}\n- Description: {video_info.get('description', '')[:300]}\nTask: Identify context and provide safe offsets."

        raw = await self.llm.agenerate(system_prompt=sys_prompt, user_prompt=user_prompt, media_files=media_files)
        res = self._parse_llm_json(raw) or {}
        return res.get('identity_grounding', 'Unknown'), res.get('segment_analysis', {})

    async def _step2_kg_walk(self, identity_data, category):
        search_candidates = []
        if isinstance(identity_data, dict):
            if identity_data.get('wikidata_search_anchor'): search_candidates.append(identity_data['wikidata_search_anchor'])
            if identity_data.get('full_identity'): search_candidates.append(identity_data['full_identity'].split(',')[0].split(':')[-1].strip())
        else:
            search_candidates.append(str(identity_data))

        path_result = None
        for candidate in search_candidates:
            query = re.sub(r'\(.*?\)', '', candidate).strip()
            path_result = await self.kg_walker.walk(start_title=query, steps=5)
            if path_result: break 

        if not path_result: return None
            
        raw_path = path_result["path"]
        nodes_info = {n["title"]: n["text"] for n in raw_path}
        node_sequence = [n["title"] for n in raw_path]
        
        return {"start_entity": path_result["start_entity"], "node_sequence": node_sequence, "nodes": nodes_info}

    async def _step3_generate_deep_qa(self, category, theme, video_info, timing_info, media_files, kg_path, identity, seg_analysis):
        prompt_config = self.prompts.get(category.lower(), self.prompts.get('speech', {}))
        sys_prompt = prompt_config.get('common_system', '')

        user_msg = (
            f"CATEGORY: {category}\nAUDIO SPEAKER IDENTITY: {identity}\n"
            f"AUDIO CONTEXT: {seg_analysis.get('justification')}\n"
            f"Node Sequence: {' -> '.join(kg_path['node_sequence'])}\n"
            f"Node Article Snippets: {json.dumps(kg_path['nodes'], ensure_ascii=False)}\n"
            f"Task: Generate a multi-hop deep-search challenge."
        )

        raw = await self.llm.agenerate(system_prompt=sys_prompt, user_prompt=user_msg, media_files=media_files)
        parsed = self._parse_llm_json(raw)
        if not parsed or not parsed.get('challenges'): return

        video_url = video_info.get('url', f"https://www.youtube.com/watch?v={video_info.get('id', '')}")

        result_data = {
            "sample_id": f"{category}_{video_info['id']}_{timing_info['abs_start'].replace(':', '')}",
            "audio_file": timing_info['audio_file'],
            "video_info": {
                "title": video_info.get('title'), "url": video_url,
                "start_time": timing_info['abs_start'], "end_time": timing_info['abs_end'],
                "theme": theme, "identity": identity
            },
            "kg_info": {"start_entity": kg_path['start_entity'], "node_sequence": kg_path['node_sequence']},
            "challenge": parsed
        }

        self.final_benchmark[category].append(result_data)

    async def _search_wikipedia_title_async(self, query):
        """在线程池中执行同步的 Wikipedia API 模糊搜索"""
        def _sync():
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1}
            try:
                resp = self.session.get(search_url, params=params, timeout=5)
                data = resp.json()
                if data.get("query", {}).get("search"):
                    return data["query"]["search"][0]["title"]
            except: pass
            return query
        return await asyncio.to_thread(_sync)
    
    async def _step1_investigate_interaction(self, domain, target_entity, video_info, media_files):

        sys_prompt = self.prompts.get('interaction_investigator_system', '')
        total_dur = self.num_consecutive * self.slice_duration

        sys_prompt = sys_prompt.replace("{num_slices}", str(self.num_consecutive))
        sys_prompt = sys_prompt.replace("{slice_duration}", str(self.slice_duration))
        sys_prompt = sys_prompt.replace("{total_duration}", str(total_dur))
        sys_prompt = sys_prompt.replace("{domain}", domain)

        user_prompt = (
            f"Target Entity to Find: {target_entity}\n"
            f"Known Context:\n- Title: {video_info.get('title')}\n- Description: {video_info.get('description', '')[:300]}\n"
            f"Task: Verify the target entity and provide safe offsets."
        )
        
        raw = await self.llm.agenerate(system_prompt=sys_prompt, user_prompt=user_prompt, media_files=media_files)
        res = self._parse_llm_json(raw) or {}

        if not res.get("contains_target", False):
            return None, {}
        return res.get('acoustic_signature', 'Unknown acoustic feature'), res.get('segment_analysis', {})
    
    async def _merge_clip_dynamic(self, pair_id, video_id, wav_paths, base_sec, seg_analysis, label):
        safe_start = float(seg_analysis.get('safe_start_offset_seconds', 0))
        safe_end = float(seg_analysis.get('safe_end_offset_seconds', 10.0))
        
        abs_start = base_sec - self.half_duration + safe_start
        abs_end = abs_start + (safe_end - safe_start)
        start_time_str = self._seconds_to_hms(abs_start).replace(':', '')
        
        group_dir = os.path.join(self.media_output_dir, pair_id)
        os.makedirs(group_dir, exist_ok=True)
        
        filename = f"INTER_{label}_{video_id}_{start_time_str}.wav"
        output_path = os.path.join(group_dir, filename)

        success = await self._crop_and_merge_audio(wav_paths, safe_start, safe_end, output_path)
        
        if success:
            return output_path, {
                "abs_start": self._seconds_to_hms(abs_start),
                "abs_end": self._seconds_to_hms(abs_end),
                "file": os.path.join("audio_clips", pair_id, filename)
            }
        return None, None
    
    async def _process_interaction_pair(self, pair_id, pair_dir, pair_meta):
        """核心处理逻辑：支持 N 个 side 的动态处理"""
        async with self.semaphore:
            side_labels = sorted(pair_meta['sides'].keys())
            processed_clips = []
            
            for label in side_labels:
                is_side_success = False
                
                for video_info in pair_meta['sides'][label]:
                    side_subdir = os.path.join(pair_dir, label)
                    timed_files = await self._prepare_interaction_audio_async(video_info, side_subdir)
                    if not timed_files: 
                        continue 
                    
                    pick_n = min(self.num_consecutive, len(timed_files))
                    max_start_idx = max(1, len(timed_files) - pick_n + 1)
                    available_indices = list(range(0, max_start_idx, pick_n))
                    random.shuffle(available_indices)
                    
                    domain = video_info.get('domain', 'SPEECH')
                    entity_name = pair_meta['pair_info'].get(f'entity_{label}', 'Unknown')

                    for attempt in range(min(3, len(available_indices))):
                        start_idx = available_indices[attempt]
                        selected_items = timed_files[start_idx: start_idx + pick_n]
                        
                        sample_folder_path = os.path.join(side_subdir, f"samples_{video_info['id']}")
                        slices = [os.path.abspath(os.path.join(sample_folder_path, x['file'])) for x in selected_items]
                        base_sec = selected_items[0]['sec']

                        try:
                            acoustic_signature, seg = await self._step1_investigate_interaction(
                                domain, entity_name, video_info, slices
                            )
                            
                            if not acoustic_signature or not seg.get('is_safe'): 
                                continue

                            final_path, timing = await self._merge_clip_dynamic(pair_id, video_info['id'], slices, base_sec, seg, label)
                            if not final_path: continue 

                            if not await self._audit_audio_content(domain, final_path):
                                if os.path.exists(final_path): os.remove(final_path)
                                continue 

                            processed_clips.append({
                                "label": label,
                                "video_info": video_info,
                                "entity": entity_name,          
                                "acoustic_signature": acoustic_signature, 
                                "final_path": final_path,
                                "timing": timing,
                                "domain": domain
                            })
                            is_side_success = True
                            break

                        except Exception as e:
                            logger.warning(f"    [Interaction Attempt {attempt+1}/3 Error] Side {label}: {e}")
                            continue
                    
                    if is_side_success:
                        break
                
                if not is_side_success:
                    logger.warning(f"      [Skip] Side {label} failed all candidate videos in {pair_id}")
                    return

            try:
                bridge_node = pair_meta.get('pair_info', {}).get('intersection_entity')
                if not bridge_node: return
                
                query = re.sub(r'\(.*?\)', '', bridge_node.strip().strip('"').strip("'")).strip() 
                corrected_node = await self._search_wikipedia_title_async(query)
                path_result = await self.kg_walker.walk(start_title=corrected_node, steps=5)
                if not path_result: return

                kg_path = {
                    "start_entity": path_result["start_entity"],
                    "node_sequence": [n["title"] for n in path_result["path"]],
                    "nodes": {n["title"]: n["text"] for n in path_result["path"]}
                }

                sys_gen = self.prompts.get('interaction_generator', {}).get('system', '')
                identities_text = ""
                media_files = []
                video_info_list = []
                
                for idx, clip in enumerate(processed_clips):
                    i = idx + 1
                    identities_text += f"--- CLIP {i} ({clip['entity']}) ---\nAcoustic Signature: {clip['acoustic_signature']}\n\n"
                    media_files.append(clip['final_path'])

                    video_info_list.append({
                        "id": clip['video_info']['id'],
                        "side": clip['label'],
                        "domain": clip['domain'],
                        "entity": clip['entity'],   
                        "start_time": clip['timing']['abs_start']
                    })

                usr_gen = (
                    f"{identities_text}"
                    f"--- BRIDGE ENTITY ---\n{corrected_node}\n\n"
                    f"--- NODE SEQUENCE ---\n{' -> '.join(kg_path['node_sequence'])}\n\n"
                    f"--- NODE CONTEXT ---\n{json.dumps(kg_path['nodes'], ensure_ascii=False)}\n\n"
                    f"Task: Generate a challenge that REQUIRES listening to ALL {len(media_files)} clips."
                )
                
                raw_gen = await self.llm.agenerate(sys_gen, usr_gen, media_files=media_files)
                parsed_challenge = self._parse_llm_json(raw_gen)
                if not parsed_challenge or not parsed_challenge.get('challenges'): return
                
                result_data = {
                    "sample_id": f"INTER_{pair_id}_{len(media_files)}WAY",
                    "audio_files": [c['timing']['file'] for c in processed_clips],
                    "video_info": {
                        "identity_summary": " | ".join([f"Clip {c['label']} ({c['entity']}): {c['acoustic_signature']}" for c in processed_clips]),
                        "interaction_context": pair_meta['pair_info'].get('connection'),
                        "clips": video_info_list
                    },
                    "kg_info": {"start_entity": kg_path['start_entity'], "node_sequence": kg_path['node_sequence']},
                    "challenge": parsed_challenge
                }

                self.final_benchmark["INTERACTION"].append(result_data)
                
            except Exception as e:
                logger.error(f"    [Error] Interaction failed for {pair_id}: {e}")

    async def _prepare_interaction_audio_async(self, video_info, side_dir):
        """音频预处理逻辑异步包装 - 修改为只返回全部切片列表"""
        def _sync():
            video_id = video_info['id']
            sample_folder_path = os.path.join(side_dir, f"samples_{video_id}")
            if not os.path.exists(sample_folder_path): return None

            all_files = [f for f in os.listdir(sample_folder_path) if f.endswith('.wav')]
            timed_files = sorted(
                [{"sec": self._parse_timestamp_to_seconds(re.search(r'_(?P<ts>\d+m\d+s)\.wav$', f).group('ts')),
                  "ts_str": re.search(r'_(?P<ts>\d+m\d+s)\.wav$', f).group('ts'), "file": f}
                 for f in all_files if re.search(r'_(?P<ts>\d+m\d+s)\.wav$', f)], key=lambda x: x['sec'])
            
            if not timed_files: return None

            return timed_files
        
        return await asyncio.to_thread(_sync)

    async def run_interaction_generation(self):
        """交互题异步并行调度 - 兼容 2 音频及多音频"""
        if not os.path.exists(self.interaction_dir): 
            logger.warning(f"Interaction dir {self.interaction_dir} does not exist. Writing empty benchmark.")
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.final_benchmark, f, indent=2, ensure_ascii=False)
            return
        
        pairs = [d for d in os.listdir(self.interaction_dir) 
                 if os.path.isdir(os.path.join(self.interaction_dir, d))]

        tasks = []
        for pair_id in pairs:
            pair_path = os.path.join(self.interaction_dir, pair_id)
            meta_path = os.path.join(pair_path, "pair_meta.json")
            if not os.path.exists(meta_path): continue

            with open(meta_path, 'r', encoding='utf-8') as f:
                pair_meta = json.load(f)
            tasks.append(self._process_interaction_pair(pair_id, pair_path, pair_meta))

        if tasks:
            await tqdm.gather(*tasks, desc="Generating Interactions")
            
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.final_benchmark, f, indent=2, ensure_ascii=False)

    async def run_full_pipeline(self):
        """全量流水线异步驱动"""
        logger.info("========== ASYNC PIPELINE START ==========")

        logger.info(f"[Step 1] Generating raw questions (Async)...")
        await self.run()
        logger.info(f"[Step 1] Generation complete -> {self.output_file}")

    
        self.output_file = os.path.join(self.run_dir, "benchmark.json")
        self.media_output_dir = os.path.join(self.run_dir, "audio_clips")
        self.filtered_1_file = os.path.join(self.run_dir, "benchmark_filtered.json")
        self.rejected_1_file = os.path.join(self.run_dir, "benchmark_rejected.json")
        self.final_hard_file = os.path.join(self.run_dir, "benchmark_FINAL_HARD.json")
        self.still_easy_file = os.path.join(self.run_dir, "benchmark_STILL_EASY.json")

        refiner_engine = BenchmarkRefiner()
        necessity_tester = AudioNecessityTester()

        logger.info(f"[Step 3 & 4] Executing Refinement & Second Filter (Async)...")
        await refiner_engine.process_single_file(
            input_file=self.output_file, 
            output_hard_file=self.final_hard_file, 
            output_easy_file=self.still_easy_file
        )

        logger.info("[Step 3] Executing Audio Necessity Testing (Blind Search)...")
        if os.path.exists(self.final_hard_file):
            await necessity_tester.process_json_file(self.final_hard_file)
            
    async def run_interaction_full_pipeline(self):
            await self.run_interaction_generation()  

            refiner_engine = BenchmarkRefiner()
            await refiner_engine.process_single_file(
                input_file=self.output_file,
                output_hard_file=self.final_hard_file,
                output_easy_file=self.still_easy_file
            )
            necessity_tester = AudioNecessityTester()
            if os.path.exists(self.final_hard_file):
                await necessity_tester.process_json_file(self.final_hard_file)



if __name__ == "__main__":
    pipeline = AudioDeepSearchPipeline(is_interaction=True)

    asyncio.run(pipeline.run_full_pipeline())

    # asyncio.run(pipeline.run_interaction_full_pipeline())

