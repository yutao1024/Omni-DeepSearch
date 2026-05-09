import os
import json
import yaml
import logging
import re
import asyncio
from datetime import datetime
from llm_provider import get_llm_provider


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "data_workspace", "benchmark_runs")
FILTERED_BASE_DIR = os.path.join(BASE_DIR, "data_workspace", "filtered_runs")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Filter")

class BenchmarkFilter:
    def __init__(self, config_path="config/config.yaml", prompt_path="config/prompts.yaml", max_concurrent=10):
        with open(os.path.join(BASE_DIR, config_path), 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(os.path.join(BASE_DIR, prompt_path), 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        
        self.llm = get_llm_provider(self.config['llm'], self.config['llm']['filter_provider'])
        
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _parse_llm_json(self, text):
        try:
            return json.loads(text)
        except:
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1:
                    return json.loads(text[start:end+1])
            except Exception as e:
                logger.error(f"      [Critical] Failed to parse JSON even with regex: {e}")
        return None

    async def _assess_single_item(self, category, item):
        async with self.semaphore:
            try:
                video_info = item.get('video_info', {})
                if category == "INTERACTION":
                    identity = video_info.get('identity_summary', 'Unknown')
                else:
                    identity = video_info.get('identity', 'Unknown')

                challenge = item['challenge']['challenges'][0]
                question = challenge['question']
                ground_truth = challenge['ground_truth_answer']

                baseline_sys = self.prompts['quality_assessment']['baseline_system']
                baseline_user = f"Source Identity: {identity}\nDeep Search Challenge: {question}"
                baseline_res = await self.llm.agenerate(system_prompt=baseline_sys, user_prompt=baseline_user)

                assessor_sys = self.prompts['quality_assessment']['assessor_system']
                assessor_user = f"Source Identity: {identity}\n Question: {question}\nGround Truth: {ground_truth}\nBaseline: {baseline_res}"
                audit_raw = await self.llm.agenerate(system_prompt=assessor_sys, user_prompt=assessor_user)
                
                audit_json = self._parse_llm_json(audit_raw)
                is_hard_enough = audit_json.get('is_challenging', False) if audit_json else False
                
                if is_hard_enough:
                    logger.info(f"    [KEEP] Valid Challenge: {item['sample_id']}")
                    return item, True
                else:
                    logger.info(f"    [FILTER] {item['sample_id']}")
                    return item, False
            except Exception as e:
                logger.error(f"    [Audit Error] {item.get('sample_id')}: {e}")
                return item, False

    async def process_single_file(self, input_file, output_kept_file, output_rejected_file):

        if not os.path.exists(input_file):
            logger.warning(f"Filter input file missing: {input_file}")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filtered_data = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}
        rejected_data = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}

        for category in filtered_data.keys():
            items = data.get(category, [])
            if not items: continue
            
            logger.info(f"    Filtering {category} ({len(items)} items)...")
            
            tasks = [self._assess_single_item(category, item) for item in items]
            results = await asyncio.gather(*tasks)
            
            for result_item, is_kept in results:
                if is_kept:
                    filtered_data[category].append(result_item)
                else:
                    rejected_data[category].append(result_item)



        os.makedirs(os.path.dirname(output_kept_file), exist_ok=True)
        with open(output_kept_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        with open(output_rejected_file, 'w', encoding='utf-8') as f:
            json.dump(rejected_data, f, indent=2, ensure_ascii=False)
        
        total_kept = sum(len(v) for v in filtered_data.values())
        total_rejected = sum(len(v) for v in rejected_data.values())
        logger.info(f"    Filter Results: Kept {total_kept}, Rejected {total_rejected}")

    async def process_all_runs(self):
        if not os.path.exists(RUNS_DIR):
            logger.error("No benchmark_runs directory found.")
            return

        run_folders = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
        
        for folder in run_folders:
            input_file = os.path.join(RUNS_DIR, folder, "benchmark.json")
            if not os.path.exists(input_file):
                continue

            logger.info(f">>> Filtering Run: {folder}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filtered_data = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}
            rejected_data = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}

            for category in filtered_data.keys():
                items = data.get(category, [])
                if not items: continue
                
                logger.info(f"    Processing {category} ({len(items)} items)...")
                
                tasks = [self._assess_single_item(category, item) for item in items]
                results = await asyncio.gather(*tasks)
                
                for result_item, is_kept in results:
                    if is_kept:
                        filtered_data[category].append(result_item)
                    else:
                        rejected_data[category].append(result_item)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(FILTERED_BASE_DIR, f"filtered_{folder}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, "benchmark_filtered.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            rejected_file = os.path.join(output_dir, "benchmark_rejected.json")
            with open(rejected_file, 'w', encoding='utf-8') as f:
                json.dump(rejected_data, f, indent=2, ensure_ascii=False)
            
            total_kept = sum(len(v) for v in filtered_data.values())
            logger.info(f"    Done for {folder}: Kept {total_kept}")

    async def process_custom_file(self, input_file_path, output_tag="refined"):
        """
        处理自定义文件。逻辑完全保留。
        """
        if not os.path.exists(input_file_path):
            logger.error(f"File not found: {input_file_path}")
            return

        logger.info(f">>> Re-Filtering File: {input_file_path}")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filtered_data = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}
        
        for category, items in data.items():
            if not items: continue
            
            tasks = [self._assess_single_item(category, item) for item in items]
            results = await asyncio.gather(*tasks)
            
            for result_item, is_kept in results:
                if is_kept:
                    filtered_data[category].append(result_item)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(FILTERED_BASE_DIR, f"final_check_{output_tag}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "benchmark_filtered_final.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Final Filtered Result: {output_file}")


if __name__ == "__main__":
    filter_engine = BenchmarkFilter()
    asyncio.run(filter_engine.process_all_runs())