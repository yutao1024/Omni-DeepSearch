import os
import json
import logging
import yaml
import re
import asyncio
from datetime import datetime
from wiki_utils import download_and_resize_image
from audio_qa_pipeline import AudioDeepSearchPipeline
from enhanced_wiki_walker import EnhancedWikiWalker

from benchmark_filter import BenchmarkFilter
from audio_necessity_tester import AudioNecessityTester
from visual_necessity_tester import VisualNecessityTester 

logger = logging.getLogger("AudioToImagePipeline")

class AudioToImagePipeline(AudioDeepSearchPipeline):
    def __init__(self, config_path="config/config.yaml", prompt_path="config/prompts.yaml"):
        current_base_dir = os.path.dirname(os.path.abspath(__file__))
        abs_config_path = os.path.normpath(os.path.join(current_base_dir, config_path))
        abs_prompt_path = os.path.normpath(os.path.join(current_base_dir, prompt_path))
        
        super().__init__(config_path=abs_config_path, prompt_path=abs_prompt_path)

        self.config_path_abs = abs_config_path
        self.prompt_path_abs = abs_prompt_path

        self.run_dir = self._create_fresh_run_dir()
        self.output_file = os.path.join(self.run_dir, "benchmark_audio_to_image.json")
        self.media_output_dir = os.path.join(self.run_dir, "audio_clips")
        
        self.temp_image_cache = os.path.join(self.run_dir, "target_images_for_llm")
        os.makedirs(self.temp_image_cache, exist_ok=True)
        os.makedirs(self.media_output_dir, exist_ok=True)

        self.final_benchmark = {
            "SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], 
            "AUDIO_TO_IMAGE": [] 
        }
        
        self.kg_walker = EnhancedWikiWalker(llm_client=self.llm, llm_model="Not_Used", config_path=abs_prompt_path)
        self.kg_walker.audit_storage_dir = self.temp_image_cache

    def _create_fresh_run_dir(self,suffix=""):
        base_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_workspace", "benchmark_runs")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_A2I')
        new_dir = os.path.join(base_runs_dir, timestamp)
        os.makedirs(new_dir, exist_ok=True)
        logger.info(f"[*] Started Fresh Audio-to-Image Run: {new_dir}")
        return new_dir

    async def _step2_kg_walk(self, identity_data, category):
        if isinstance(identity_data, dict):
            start_query = identity_data.get('wikidata_search_anchor', 'Unknown')
        else:
            start_query = str(identity_data)

        if start_query == "Unknown": return None

        self.kg_walker.storage_dir = self.temp_image_cache 

        path_result = await self.kg_walker.walk_with_images(start_title=start_query, steps=8)
        
        if not path_result: return None
        
        node_sequence = [node["title"] for node in path_result["path"]]

        return {
            "start_entity": path_result["start_entity"],
            "path_list": path_result["path"],
            "node_sequence": node_sequence,
            "nodes": {node["title"]: node["text"] for node in path_result["path"]}
        }

    async def _step3_generate_deep_qa(self, category, theme, video_info, timing_info, media_files, 
                                     kg_path, identity, seg_analysis):
        abs_image_path = None
        try:

            best_node = next((n for n in reversed(kg_path["path_list"]) if "chosen_image" in n), None)
            
            if not best_node:
                logger.warning(f"No node with chosen_image found in path for {identity}")
                return

            target_entity = best_node["title"]
            best_img_obj = best_node["chosen_image"]
            
            abs_image_path = best_img_obj.get('local_path')
            best_image_url = best_img_obj.get('url')

            if not abs_image_path or not os.path.exists(abs_image_path):
                logger.error(f"Target image missing on disk: {abs_image_path}")
                return

            img_filename = os.path.basename(abs_image_path)
            generator_cfg = self.prompts.get('audio_to_image_generator', {})
            user_tpl = generator_cfg.get('user_template', "")

            final_user_msg = (
                user_tpl.replace("{{identity}}", str(identity))
                .replace("{{justification}}", seg_analysis.get('justification', ''))
                .replace("{{node_sequence}}", " -> ".join(kg_path['node_sequence'])) 
                .replace("{{nodes}}", json.dumps(kg_path['nodes'], ensure_ascii=False))
                .replace("{{target_entity}}", target_entity) 
            )

            logger.info(f"Generating challenges for {target_entity} using image: {img_filename}")
            raw_challenge = await self.llm.agenerate(
                system_prompt=generator_cfg.get('system', ""),
                user_prompt=final_user_msg,
                media_files=media_files + [abs_image_path]
            )
            
            parsed = self._parse_llm_json(raw_challenge)
            
            if not parsed or not parsed.get('challenges'):
                logger.warning(f"LLM failed to generate valid QA for {target_entity}. Cleaning up image.")
                if abs_image_path and os.path.exists(abs_image_path):
                    await asyncio.to_thread(os.remove, abs_image_path)
                return

            sample_id = f"A2I_{video_info['id']}_{timing_info['abs_start'].replace(':', '')}"
            relative_visual_path = os.path.join("target_images_for_llm", img_filename)
            inner_qa = parsed['challenges'][0]
            video_info_payload  = video_info.copy()
            video_info_payload['identity'] = kg_path['start_entity'] 
            self.final_benchmark["AUDIO_TO_IMAGE"].append({
                "sample_id": sample_id,
                "audio_file": timing_info['audio_file'], 
                "visual_file": relative_visual_path,    
                "video_info": video_info_payload,
                "kg_info": {
                    "start_entity": kg_path['start_entity'],
                    "node_sequence": kg_path['node_sequence']
                },
                "challenge": {
                    "challenges": [{
                        "question": inner_qa.get('question'),
                        "reasoning_checklist": inner_qa.get('reasoning_checklist', []),
                        "ground_truth_answer": inner_qa.get('ground_truth_answer')
                    }]
                },
                "visual_meta": {
                    "reference_image_url": best_image_url,
                    "target_entity": target_entity
                }
            })
            
            logger.info(f"[SUCCESS] {sample_id} created with target {target_entity}.")

        except Exception as e:
            logger.error(f"Unexpected error in _step3 for {identity}: {e}")
            if abs_image_path and os.path.exists(abs_image_path):
                 await asyncio.to_thread(os.remove, abs_image_path)

    async def run_pipeline(self):
        logger.info("========== STEP 1: AUDIO-TO-IMAGE GENERATION ==========")
        await self.run() 
        
        final_output = {"AUDIO_TO_IMAGE": self.final_benchmark["AUDIO_TO_IMAGE"]}
        def _save_raw():
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
        await asyncio.to_thread(_save_raw)
        
        logger.info(f"Raw Generation Finished: {self.output_file}")

        logger.info("========== STEP 2: QUALITY FILTERING ==========")
        filter_engine = BenchmarkFilter()
        
        filtered_path = self.output_file.replace(".json", "_FILTERED.json")
        rejected_path = self.output_file.replace(".json", "_REJECTED.json")
        
        await self._execute_custom_filter(filter_engine, self.output_file, filtered_path, rejected_path)

        if os.path.exists(filtered_path):
            logger.info("========== STEP 3: AUDIO NECESSITY TESTING ==========")
            tester = AudioNecessityTester()

            await tester.process_json_file(filtered_path)
            
            final_nec_path = filtered_path.replace(".json", "_AUDIO_NECESSARY.json")
            if os.path.exists(final_nec_path):
                logger.info("========== STEP 4: VISUAL NECESSITY TESTING ==========")
                visual_tester = VisualNecessityTester(self.config, self.prompts)
                final_verified_path = visual_tester.process_json_file(final_nec_path)
                logger.info(f"DONE! Final Visual-Verified Dataset: {final_verified_path}")
        else:
            logger.warning("No items passed the quality filter. Skipping Step 3 & 4.")

    async def _execute_custom_filter(self, filter_engine, input_file, out_kept, out_rej):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.get("AUDIO_TO_IMAGE", [])
        if not items: return

        tasks = [filter_engine._assess_single_item("AUDIO_TO_IMAGE", item) for item in items]
        results = await asyncio.gather(*tasks)
        
        kept, rejected = [], []
        for processed_item, is_kept in results:
            if is_kept:
                kept.append(processed_item)
            else:
                rejected.append(processed_item)

        def _save_filtered():
            with open(out_kept, 'w', encoding='utf-8') as f:
                json.dump({"AUDIO_TO_IMAGE": kept}, f, indent=2, ensure_ascii=False)
            with open(out_rej, 'w', encoding='utf-8') as f:
                json.dump({"AUDIO_TO_IMAGE": rejected}, f, indent=2, ensure_ascii=False)
        
        await asyncio.to_thread(_save_filtered)
        logger.info(f"Filter Summary: {len(kept)} Kept, {len(rejected)} Rejected.")

if __name__ == "__main__":
    pipeline = AudioToImagePipeline()
    asyncio.run(pipeline.run_pipeline())