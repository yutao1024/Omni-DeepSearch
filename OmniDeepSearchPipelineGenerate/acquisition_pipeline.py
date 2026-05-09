import os
import json
import yaml
import re
import asyncio
import logging
import shutil
import time
from audio_to_image_pipeline import AudioToImagePipeline
from video_tools import VideoTools
from llm_provider import get_llm_provider
from audio_qa_pipeline import AudioDeepSearchPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_logging(workspace):
    log_dir = os.path.join(workspace, "logs")
    os.makedirs(log_dir, exist_ok=True)

    p_logger = logging.getLogger("Acquisition")
    p_logger.setLevel(logging.INFO)
    p_fh = logging.FileHandler(os.path.join(log_dir, "acquisition.log"), encoding='utf-8')
    p_fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    p_logger.addHandler(p_fh)
    p_logger.addHandler(logging.StreamHandler())

    llm_logger = logging.getLogger("LLM_Chat")
    llm_logger.setLevel(logging.INFO)
    l_fh = logging.FileHandler(os.path.join(log_dir, "llm_chat.log"), encoding='utf-8')
    l_fh.setFormatter(logging.Formatter('%(asctime)s\n%(message)s\n' + '=' * 60 + '\n'))
    llm_logger.addHandler(l_fh)

    return p_logger, llm_logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR) 
class AcquisitionPipeline:
    def __init__(self, config_path="config/config.yaml"):
        config_path = os.path.join(BASE_DIR, config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        p_path = self.config.get('prompts_path') or self.config['acquisition'].get('prompts_path')
        p_path = os.path.join(BASE_DIR, p_path)
        with open(p_path, 'r', encoding='utf-8') as f:
            self.prompts_db = yaml.safe_load(f)

        self.workspace = os.path.join(BASE_DIR, self.config['acquisition']['workspace'])
        self.p_logger, self.l_logger = setup_logging(self.workspace)

        self.vt = VideoTools()
        self.llm = get_llm_provider(self.config['llm'], self.config['llm']['default_provider'])

        self.db_id_path = os.path.join(self.workspace, "downloaded_ids.json")
        self.db_theme_path = os.path.join(self.workspace, "processed_themes.json")
        self.global_metadata_path = os.path.join(self.workspace, "all_videos_summary.json")
        self.db_inter_path = os.path.join(self.workspace, "processed_interactions.json")

        self.processed_interactions = self._load_json_sync(self.db_inter_path, set)
        self.downloaded_ids = self._load_json_sync(self.db_id_path, set)
        self.processed_themes = self._load_json_sync(self.db_theme_path, set)
        self.global_metadata = self._load_json_sync(self.global_metadata_path, list)

        self.semaphore = asyncio.Semaphore(self.config.get('max_threads', 10))
        self.theme_semaphore = asyncio.Semaphore(10)
        self.io_lock = asyncio.Lock()

    def _load_json_sync(self, path, container_type):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return container_type(json.load(f))
        return container_type()

    async def _save_json_async(self, path, data):
        async with self.io_lock:
            def _write():
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(list(data), f, ensure_ascii=False, indent=2)
            await asyncio.to_thread(_write)

    def sanitize_name(self, name):
        return re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))

    def _extract_json(self, text):
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            text = text.strip()
            try:
                return json.dumps(json.loads(text))
            except json.JSONDecodeError:
                pass
            
            all_pairs = []
            blocks = re.findall(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if blocks:
                for b in blocks:
                    try:
                        data = json.loads(b)
                        if isinstance(data, list): all_pairs.extend(data)
                    except: continue
                return json.dumps(all_pairs)
            
            obj_match = re.search(r'\{.*\}', text, re.DOTALL)
            if obj_match: return f"[{obj_match.group()}]"
            return text
        except: return "[]"

    def _log_llm(self, task, sys, user, res):
        self.l_logger.info(f"TASK: {task}\n[SYS]: {sys}\n[USER]: {user}\n[RES]: {res}")

    async def _perform_slicing_async(self, vid_id, video_path, theme_dir):
        """适配异步 VideoTools 的切片操作"""
        await self.vt.process_multimodal_sampling(
            video_manifest={vid_id: {"path": video_path}},
            interval=self.config['sampling']['interval'],
            workspace=theme_dir,
            need_audio=True,
            audio_duration=self.config['sampling']['audio_duration']
        )

        def _sync_rename():
            src_path = os.path.join(theme_dir, "samples", vid_id)
            dst_path = os.path.join(theme_dir, f"samples_{vid_id}")
            if os.path.exists(src_path):
                if os.path.exists(dst_path): shutil.rmtree(dst_path)
                os.rename(src_path, dst_path)
            shutil.rmtree(os.path.join(theme_dir, "samples"), ignore_errors=True)
            shutil.rmtree(os.path.join(theme_dir, "audio_temp"), ignore_errors=True)
        
        await asyncio.to_thread(_sync_rename)

    async def run(self):
        self.p_logger.info("=== Starting Strategic Acquisition Pipeline (Async) ===")
        if self.processed_themes:
            theme_exclusion_list = "\n".join([f"- {t}" for t in self.processed_themes])
        else:
            theme_exclusion_list = "No previously processed themes."
        self.p_logger.info(f"[Theme Filter] Loaded {len(self.processed_themes)} processed single themes, will exclude for LLM generation.")

        for domain in self.config['domains']:
            domain_id = domain['id']
            specific_rule = self.prompts_db['theme_brainstormer']['rules'].get(domain_id, "")
            sys_p = self.prompts_db['theme_brainstormer']['system']
            user_p = (self.prompts_db['theme_brainstormer']['user_template']
                      .replace("{{name}}", domain['name'])
                      .replace("{{count}}", str(self.config['acquisition']['themes_per_domain']))
                      .replace("{{specific_rule}}", specific_rule)
                      .replace("{{processed_themes_exclusion}}", theme_exclusion_list))

            res = await self.llm.agenerate(sys_p, user_p)
            self._log_llm(f"Brainstorm-{domain['id']}", sys_p, user_p, res)

            try:
                res_clean = res.split("```json")[1].split("```")[0].strip() if "```json" in res else res.strip()
                themes = json.loads(res_clean)
            except: continue

            async def _process_single_theme(theme):
                async with self.theme_semaphore:
                    folder_name = self.sanitize_name(theme)
                    if not folder_name or folder_name in self.processed_themes: return

                    save_dir = os.path.join(self.workspace, "downloads", domain['id'], folder_name)
                    os.makedirs(save_dir, exist_ok=True)

                    q_sys = self.prompts_db['query_generator']['system'].replace("{{theme}}", theme)
                    q_user = (self.prompts_db['query_generator']['user_template']
                            .replace("{{theme}}", theme)
                            .replace("{{domain_name}}", domain['name'])
                            .replace("{{count}}", str(self.config['acquisition'].get('queries_per_theme', 1))))
                    
                    q_res = await self.llm.agenerate(q_sys, q_user)
                    try:
                        q_res_clean = q_res.split("```json")[1].split("```")[0].strip() if "```json" in q_res else q_res.strip()
                        queries = json.loads(q_res_clean)
                    except: return

                    theme_meta_data = {"theme": theme, "domain": domain['name'], "videos": []}
                    self.p_logger.info(f"  [NEW THEME] {theme} ({domain['id']})")

                    async def _handle_single_query(query_text):
                        async with self.semaphore: 
                            stats = {"results": 0, "success": 0, "fail": 0}
                            try:
                                results = await self.vt.search_videos(query_text, self.config, max_results=self.config['acquisition']['videos_per_query'])
                                if not results: return
                                
                                stats["results"] = len(results)

                                for v in results:
                                    if v['id'] in self.downloaded_ids: continue
                                    
                                    try:
                                        manifest = await self.vt.download_videos([v['url']], save_dir, self.config, height=720)
                                        if not manifest or v['id'] not in manifest: continue
                                        
                                        video_path = manifest[v['id']]['path']
                                        self.downloaded_ids.add(v['id'])
                                        
                                        full_meta = await self.vt.fetch_video_metadata(v['url'], self.config)
                                        await self._perform_slicing_async(v['id'], video_path, save_dir)
 
                                        try:
                                            if os.path.exists(video_path):
                                                os.remove(video_path)
                                                self.p_logger.info(f"      [Cleanup] Deleted video file: {video_path}")
                                        except Exception as e:
                                            self.p_logger.warning(f"      [Cleanup Failed] Could not delete {video_path}: {e}")

                                        video_entry = {
                                            "id": v['id'], "title": full_meta.get('full_title', v.get('title', 'Unknown')),
                                            "url": v['url'], "theme": theme, 
                                            "description": full_meta.get('full_description', 'No description.'),
                                            "download_time": time.strftime("%Y-%m-%d %H:%M:%S")
                                        }
                                        
                                        theme_meta_data["videos"].append(video_entry)
                                        self.global_metadata.append(video_entry)
                                        
    
                                        await self._save_json_async(self.global_metadata_path, self.global_metadata)
                                        
                                        stats["success"] += 1
                                        self.p_logger.info(f"      [OK] Downloaded {v['id']}")
                                    except Exception as e:
                                        stats["fail"] += 1
                                        self.p_logger.error(f"      [VIDEO ERROR] {v['id']} | {str(e)}")
                            except Exception as e:
                                self.p_logger.error(f"    [SEARCH ERROR] {query_text} | {str(e)}")

                    await asyncio.gather(*[_handle_single_query(q) for q in queries])

                    if theme_meta_data["videos"]:
                        with open(os.path.join(save_dir, "theme_meta.json"), 'w', encoding='utf-8') as f:
                            json.dump(theme_meta_data, f, ensure_ascii=False, indent=2)

                    self.processed_themes.add(folder_name)
                    await self._save_json_async(self.db_id_path, self.downloaded_ids)
                    await self._save_json_async(self.db_theme_path, self.processed_themes)

            tasks = [_process_single_theme(theme) for theme in themes]
            if tasks:
                await asyncio.gather(*tasks)

        self.p_logger.info("=== Acquisition Pipeline Finished ===")

    async def run_interaction_pipeline(self):
        self.p_logger.info("=== Starting Interaction Acquisition Pipeline (Async, Multi-Side Support) ===")
        inter_cfg = self.config['acquisition'].get('interaction', {})
        pairs_count = inter_cfg.get('pairs_to_generate', 5)
        audio_count = inter_cfg.get('audio_count', 3) 
        v_per_side = inter_cfg.get('videos_per_side', 1)

        base_inter_dir = os.path.join(BASE_DIR, self.config.get('interaction_dir', 'data_workspace/interactions'))

        if not os.path.exists(base_inter_dir):
            os.makedirs(base_inter_dir)
        current_count_exclusions = []
    
        for item in self.processed_interactions:
            entity_count_in_id = len(item.split('_and_'))
            if entity_count_in_id == audio_count:
                clean_name = item.replace("group_", "").replace("_and_", " + ").replace("_", " ")
                current_count_exclusions.append(f"- {clean_name}")

        exclusion_list_str = "\n".join(current_count_exclusions) if current_count_exclusions else "None yet for this audio count."
        self.p_logger.info(f"  [Filter] Found {len(current_count_exclusions)} existing groups for {audio_count}-audio task.")

        sys_p = self.prompts_db['interaction_brainstormer']['system']

        if audio_count == 2:
            dynamic_example = """- [MUSIC + ENV]: Hans Zimmer (MUSIC) & Supermarine Spitfire (ENV). 
      **Intersection Entity: "Dunkirk (2017 film)"** (The specific production where the engine sound was heavily integrated into the score)."""
            dynamic_json = '          {"name": "Entity 1", "domain": "MUSIC"},\n          {"name": "Entity 2", "domain": "ENV"}'
            
        elif audio_count == 3:
            dynamic_example = """- [SPEECH + BIO + ENV]: David Attenborough (SPEECH) & Mountain Gorilla (BIO) & Land Rover Defender (ENV). 
      **Intersection Entity: "Life on Earth (TV series)"** (The famous encounter in Rwanda where this vehicle was used)."""
            dynamic_json = '          {"name": "Entity 1", "domain": "SPEECH"},\n          {"name": "Entity 2", "domain": "BIO"},\n          {"name": "Entity 3", "domain": "ENV"}'
            
        elif audio_count == 4:
            dynamic_example = """- [SPEECH + MUSIC + BIO + ENV]: Leonardo DiCaprio (SPEECH) & Ryuichi Sakamoto (MUSIC) & Grizzly Bear (BIO) & Kentucky Flintlock Rifle (ENV). 
      **Intersection Entity: "The Revenant (2015 film)"** (DiCaprio acted, Sakamoto composed the score, featuring the famous bear attack and period rifle sounds)."""
            dynamic_json = '          {"name": "Entity 1", "domain": "SPEECH"},\n          {"name": "Entity 2", "domain": "MUSIC"},\n          {"name": "Entity 3", "domain": "BIO"},\n          {"name": "Entity 4", "domain": "ENV"}'
        else:
            dynamic_example = f"- Example with {audio_count} entities from different domains."
            dynamic_json = ",\n".join([f'          {{"name": "Entity {i+1}", "domain": "DOMAIN_{i+1}"}}' for i in range(audio_count)])

        user_p = (self.prompts_db['interaction_brainstormer']['user_template']
                  .replace("{{audio_count}}", str(audio_count))
                  .replace("{{generate_count}}", str(pairs_count))
                  .replace("{{dynamic_example}}", dynamic_example)
                  .replace("{{dynamic_json}}", dynamic_json)
                  .replace("{{exclusion_list}}", exclusion_list_str)) 

        res = await self.llm.agenerate(sys_p, user_p)
        try:
            pairs = json.loads(self._extract_json(res))
        except Exception as e:
            self.p_logger.error(f"Failed to parse interaction pairs: {e}")
            return
        async def _process_single_pair(pair):
            async with self.theme_semaphore:
                entities = pair.get('entities', [])
                if not entities: return
                pair_label = "_and_".join([self.sanitize_name(e['name']) for e in entities])
                pair_id = f"group_{pair_label}"
                
                if pair_id in self.processed_interactions: return

                pair_dir = os.path.join(base_inter_dir, pair_id)
                os.makedirs(pair_dir, exist_ok=True)

                pair_meta_data = {
                    "pair_info": {
                        "intersection_entity": pair.get('intersection_entity'),
                        "connection": pair.get('connection')
                    },
                    "sides": {} 
                }

                async def _process_side_dynamic(entity_info, side_key):
                    entity = entity_info['name']
                    domain = entity_info['domain']
                    side_dir = os.path.join(pair_dir, side_key)
                    os.makedirs(side_dir, exist_ok=True)

                    pair_meta_data["pair_info"][f"entity_{side_key}"] = entity
                    pair_meta_data["pair_info"][f"domain_{side_key}"] = domain

                    q_sys = self.prompts_db['query_generator']['system'].replace("{{theme}}", entity)
                    q_user = (self.prompts_db['query_generator']['user_template']
                                .replace("{{theme}}", entity)
                                .replace("{{domain_name}}", domain)
                                .replace("{{count}}", "3")) 
                    
                    q_res = await self.llm.agenerate(q_sys, q_user)
                    try:
                        queries = json.loads(self._extract_json(q_res))
                    except: return

                    success_count = 0
                    for q in queries:
                        if success_count >= v_per_side: break
                        
                        async with self.semaphore:
                            try:
                                results = await self.vt.search_videos(q, self.config, max_results=1)
                                if not results:
                                    self.p_logger.warning(f"      [Search Empty] No results for query: {q}")
                                    continue
                                    
                                for v in results:
                                    self.p_logger.info(f"      [Downloading] Trying to download {v['id']} for {entity}...")
                                    manifest = await self.vt.download_videos([v['url']], side_dir, self.config)
                                    
                                    if not manifest:
                                        self.p_logger.error(f"      [Download Failed] Manifest is empty for {v['url']}")
                                        continue
                                    
                                    if v['id'] not in manifest:
                                        self.p_logger.error(f"      [Download Failed] Video {v['id']} not in manifest")
                                        continue

                                    video_path = manifest[v['id']]['path']
                                    self.downloaded_ids.add(v['id'])
                                    
                                    await self._perform_slicing_async(v['id'], video_path, side_dir)
                                    full_meta = await self.vt.fetch_video_metadata(v['url'], self.config)
                                    try:
                                        if os.path.exists(video_path):
                                            os.remove(video_path)
                                            self.p_logger.info(f"      [Cleanup] Deleted interaction video: {video_path}")
                                    except Exception as e:
                                        self.p_logger.warning(f"      [Cleanup Failed] {e}")

                                    if side_key not in pair_meta_data["sides"]:
                                        pair_meta_data["sides"][side_key] = []

                                    pair_meta_data["sides"][side_key].append({
                                        "id": v['id'], 
                                        "title": full_meta.get('full_title', v['title']),
                                        "entity": entity, 
                                        "domain": domain, 
                                        "path": video_path
                                    })
                                    success_count += 1
                                    self.p_logger.info(f"      [Interaction OK] {side_key}: {entity} ({v['id']})")
                                    break 

                            except Exception as e:
                                self.p_logger.error(f"      [FATAL ERROR] Processing side {side_key} ({entity}) failed: {str(e)}")

                side_labels = [chr(97 + i) for i in range(len(entities))] 

                tasks = [_process_side_dynamic(entities[i], side_labels[i]) for i in range(len(entities))]
                await asyncio.gather(*tasks)

                if len(pair_meta_data["sides"]) == len(entities):
                    with open(os.path.join(pair_dir, "pair_meta.json"), 'w', encoding='utf-8') as f:
                        json.dump(pair_meta_data, f, ensure_ascii=False, indent=2)
                    
                    self.processed_interactions.add(pair_id)
                    await self._save_json_async(self.db_inter_path, self.processed_interactions)
                    await self._save_json_async(self.db_id_path, self.downloaded_ids)
                    self.p_logger.info(f"  [GROUP DONE] {pair_id} with {len(entities)} entities.")
                else:
                    self.p_logger.warning(f"  [GROUP INCOMPLETE] {pair_id} failed.")

        tasks = [_process_single_pair(pair) for pair in pairs]
        if tasks:
            await asyncio.gather(*tasks)

        self.p_logger.info("=== Interaction Acquisition Finished ===")


if __name__ == "__main__":
    async def main():
        acq = AcquisitionPipeline()
        await acq.run()
        pipeline = AudioToImagePipeline()
        await pipeline.run_pipeline()

        # pipeline = AudioDeepSearchPipeline()
        # await pipeline.run_full_pipeline()


        # acq = AcquisitionPipeline()
        # await acq.run_interaction_pipeline()
        # pipeline = AudioDeepSearchPipeline(is_interaction=True)
        # await pipeline.run_interaction_full_pipeline()

    asyncio.run(main())
