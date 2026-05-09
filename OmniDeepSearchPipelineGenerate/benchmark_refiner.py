import os
import json
import yaml
import logging
import re
import asyncio
import copy
from datetime import datetime
from tavily import TavilyClient
import random 
from llm_provider import get_llm_provider
from benchmark_filter import BenchmarkFilter 
from live_wiki_walker import LiveWikiWalker 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILTERED_RUNS_DIR = os.path.join(BASE_DIR, "data_workspace", "filtered_runs")
REFINED_RUNS_DIR = os.path.join(BASE_DIR, "data_workspace", "refined_runs")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Refiner-Pipeline")

import os
import json
import yaml
import logging
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from tavily import TavilyClient
from llm_provider import get_llm_provider
from benchmark_filter import BenchmarkFilter 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Refiner-Pipeline")

class BenchmarkRefiner:
    def __init__(self, config_path="config/config.yaml", prompt_path="config/prompts.yaml",max_concurrent=20):
        with open(os.path.join(BASE_DIR, config_path), 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(os.path.join(BASE_DIR, prompt_path), 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        self.llm = get_llm_provider(self.config['llm'], self.config['llm']['default_provider'])
        
        search_key = self.config.get('search_api_key')
        if not search_key:
            raise ValueError("search_api_key is missing in config.yaml")
        self.tavily = TavilyClient(api_key=search_key)

        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.wiki_walker = LiveWikiWalker(llm_client=self.llm, llm_model="Not_Used", config_path=prompt_path)
        self.filter_engine = BenchmarkFilter(config_path, prompt_path)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _get_latest_rejected_file(self):
        if not os.path.exists(FILTERED_RUNS_DIR): return None
        runs = sorted([d for d in os.listdir(FILTERED_RUNS_DIR)], reverse=True)
        for run in runs:
            p = os.path.join(FILTERED_RUNS_DIR, run, "benchmark_rejected.json")
            if os.path.exists(p): return p
        return None

    async def _execute_real_search_list(self, query):
        logger.info(f"    [Tavily] Querying: {query}")
        def _sync_search():
            return self.tavily.search(query=query, search_depth="advanced", max_results=5, days=180)

        try:
            response = await asyncio.to_thread(_sync_search)
            valid_results = []
            for r in response.get('results', []):
                res_item = f"Source: {r.get('url')}\nTitle: {r.get('title')}\nContent: {r.get('content')}"
                valid_results.append(res_item)
            return valid_results 
        except Exception as e:
            logger.error(f"    [Search Error] {e}")
            return []

    async def _extend_and_evolve_item(self, category, item):
        identity_data = item.get('video_info', {}).get('identity', {})
        display_id = identity_data.get('wikidata_search_anchor', 'Unknown') if isinstance(identity_data, dict) else str(identity_data)

        kg_info = item.get('kg_info', {})
        original_nodes = kg_info.get('node_sequence', []) 
        if not original_nodes:
            original_nodes = list(kg_info.get('nodes', {}).keys())
        if not original_nodes:
            original_nodes = [display_id]
                
        try:
            original_q = item['challenge']['challenges'][0]['question']
        except (KeyError, IndexError):
            original_q = "Please focus on the primary subject in the audio."

        news_node_idx = random.randint(0, len(original_nodes) - 1)
        current_path = copy.deepcopy(original_nodes[:news_node_idx + 1])
        current_node = current_path[-1] 
        visited_nodes = set(current_path)
        
        logger.info(f"    [Init] Original path: {original_nodes}. Truncated at '{current_node}'. Current path length: {len(current_path)}")

        filter_prompt = self.prompts.get('refiner_filter_news', {})
        generate_prompt = self.prompts.get('refiner_generate_question', {})
        if not filter_prompt or not generate_prompt:
            logger.error("    [Error] Prompts 'refiner_filter_news' or 'refiner_generate_question' not found!")
            return None

        walk_count = 0
        max_walks = 5
        news_data = None 

        while walk_count < max_walks:
            logger.info(f"    [Attempt {walk_count+1}] Searching news for Node: '{current_node}'")
            search_query = f'"{current_node}" news 2025 OR 2026'
            search_results = await self._execute_real_search_list(search_query)

            for idx, live_info in enumerate(search_results):
                logger.info(f"    [LLM Check] Testing news result {idx+1}/{len(search_results)} for '{current_node}'...")
                try:
                    user_prompt = filter_prompt['user'].format(
                        news_node=current_node,
                        live_news_info=live_info
                    )
                    
                    raw_res = await self.llm.agenerate(system_prompt=filter_prompt['system'], user_prompt=user_prompt)
                    match = re.search(r'\{.*\}', raw_res, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group())
                        
                        if parsed.get("status") != "REJECTED":
                            next_node = parsed.get("next_node", "Unknown Entity")
                            news_bridge = parsed.get("news_bridge", "")
                            logger.info(f"    [Success] News valid! '{current_node}' -> '{next_node}'")
                            news_data = (next_node, news_bridge, search_query)
                            break 
                        else:
                            logger.warning(f"    [News Rejected] {parsed.get('rejection_reason')}")
                except Exception as e:
                    logger.error(f"    [LLM Filter Error] {e}")

            if news_data:
                break 

            walk_count += 1
            if walk_count >= max_walks:
                break

            logger.info(f"    [WikiWalk] All news rejected for '{current_node}'. Walking to a new node...")
            try:
                walk_res = await self.wiki_walker.walk(start_title=current_node, steps=2)
                if walk_res and len(walk_res.get('path', [])) >= 2:
                    new_node = walk_res['path'][-1]['title']
                    if new_node in visited_nodes:
                        break 
                    current_node = new_node
                    visited_nodes.add(current_node)
                    current_path.append(current_node) 
                else:
                    break
            except Exception as e:
                logger.error(f"    [WikiWalk Error] {e}")
                break

        if not news_data:
            logger.warning(f"    [Final Skip] Failed to find valid news after {walk_count} attempts.")
            return None


        next_node, news_bridge, used_search_query = news_data
        current_path.append(next_node)
        visited_nodes.add(next_node)
        
        post_walk_contexts = [] 
        walk_current = next_node

        while len(current_path) < 5:
            logger.info(f"    [Post-Walk] Path length {len(current_path)} < 5. Walking from '{walk_current}'...")
            try:
                walk_res = await self.wiki_walker.walk(start_title=walk_current, steps=2)
                if walk_res and len(walk_res.get('path', [])) >= 2:
                    step_node = walk_res['path'][-1]
                    new_title = step_node['title']
                    new_text = step_node['text']

                    if new_title in visited_nodes:
                        logger.warning(f"    [Post-Walk] Cycle detected at '{new_title}'.")
                        break

                    walk_current = new_title
                    visited_nodes.add(walk_current)
                    current_path.append(walk_current)
                    
                    post_walk_contexts.append({
                        "source_node": current_path[-2],
                        "target_node": walk_current,
                        "context": new_text[:3000] 
                    })
                else:
                    logger.warning(f"    [Post-Walk] Dead end at '{walk_current}'.")
                    break
            except Exception as e:
                logger.error(f"    [Post-Walk Error] {e}")
                break

        if len(current_path) < 5:
            logger.warning(f"    [Final Skip] Path length {len(current_path)} is still < 5. Aborting.")
            return None

        logger.info(f"    [Generation] Generating final question for full path: {current_path}")
        try:
            user_prompt = generate_prompt['user'].format(
                category=category,
                original_question=original_q,
                original_nodes=original_nodes,
                news_node=current_node,
                next_node=next_node,
                news_bridge=news_bridge,
                post_walk_contexts=json.dumps(post_walk_contexts, indent=2),
                final_path=current_path
            )
            
            raw_res = await self.llm.agenerate(system_prompt=generate_prompt['system'], user_prompt=user_prompt)
            match = re.search(r'\{.*\}', raw_res, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                
                new_item = copy.deepcopy(item)
                new_item['challenge'] = parsed
                if 'kg_info' not in new_item:
                    new_item['kg_info'] = {}
                new_item['kg_info']['node_sequence'] = current_path
                
                new_item['challenge']['evolution_metadata'] = {
                    "original_path": original_nodes,
                    "news_node": current_node,
                    "next_node": next_node,
                    "final_path": current_path,
                    "total_nodes": len(current_path),
                    "search_query_used": used_search_query,
                    "reason": "Successfully generated >=5 node path with news bridge"
                }
                return new_item
            else:
                logger.warning(f"    [Parse Error] Generation LLM did not return JSON.")
        except Exception as e:
            logger.error(f"    [Generation Error] {e}")

        return None

    def run_pipeline(self):
        input_file = self._get_latest_rejected_file()
        self.output_dir = os.path.dirname(input_file)
        if not input_file:
            logger.error("No rejected items found.")
            return

        logger.info(f"=== Starting Refinement Pipeline on: {input_file} ===")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        refined_in_memory = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}

        for category, items in data.items():
            if not items: continue
            logger.info(f"[*] Refining Category: {category} (Total: {len(items)})")
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(self._extend_and_evolve_item, category, copy.deepcopy(item)) for item in items]
                for future in as_completed(futures):
                    res = future.result()
                    if res: 
                        refined_in_memory[category].append(res)

        logger.info("=== Starting Final Hardness Validation ===")
        
        final_hard = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}
        final_easy = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}
        
        for category, items in refined_in_memory.items():
            if not items: continue
            logger.info(f"    Verifying {category} (Total to verify: {len(items)})...")
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(self.filter_engine._assess_single_item, category, copy.deepcopy(item)) for item in items]
                for future in as_completed(futures):
                    try:
                        processed_item, is_kept = future.result()
                        if is_kept:
                            final_hard[category].append(processed_item)
                        else:
                            final_easy[category].append(processed_item)
                    except Exception as e:
                        logger.error(f"    [Audit Error] {e}")

        hard_file = os.path.join(self.output_dir, "benchmark_FINAL_HARD.json")
        with open(hard_file, 'w', encoding='utf-8') as f:
            json.dump(final_hard, f, indent=2, ensure_ascii=False)

        easy_file = os.path.join(self.output_dir, "benchmark_STILL_EASY.json")
        with open(easy_file, 'w', encoding='utf-8') as f:
            json.dump(final_easy, f, indent=2, ensure_ascii=False)

        logger.info(f"=== Pipeline Finished ===")
        logger.info(f"    - Hard Items: {sum(len(v) for v in final_hard.values())} -> {hard_file}")
        logger.info(f"    - Easy (Failed) Items: {sum(len(v) for v in final_easy.values())} -> {easy_file}")

    async def process_single_file(self, input_file, output_hard_file, output_easy_file):
        if not os.path.exists(input_file):
            logger.warning(f"Refiner input missing: {input_file}")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        refined_in_memory = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}
        tasks = []
        for category, items in data.items():
            if not items: continue
            logger.info(f"[*] Refining Category: {category} (Total: {len(items)})")

            tasks = [self._extend_and_evolve_item(category, item) for item in items]
            
            results = await asyncio.gather(*tasks)

            for res in results:
                if res:
                    refined_in_memory[category].append(res)

        logger.info("=== Starting Final Hardness Validation (2nd Filter) ===")
        
        final_hard = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}
        final_easy = {"SPEECH": [], "MUSIC": [], "ENV": [], "BIO": [], "INTERACTION": []}
        
        audit_tasks = []
        audit_meta = [] 
        for category, items in refined_in_memory.items():
            for item in items:
                audit_tasks.append(self.filter_engine._assess_single_item(category, item))
                audit_meta.append((category, item))

        if audit_tasks:
            results = await asyncio.gather(*audit_tasks)
            for (cat, original_item), (processed_item, is_kept) in zip(audit_meta, results):
                if is_kept:
                    final_hard[cat].append(processed_item)
                else:
                    final_easy[cat].append(processed_item)
        with open(output_hard_file, 'w', encoding='utf-8') as f:
            json.dump(final_hard, f, indent=2, ensure_ascii=False)

        with open(output_easy_file, 'w', encoding='utf-8') as f:
            json.dump(final_easy, f, indent=2, ensure_ascii=False)

        logger.info(f"    - Hard Items: {sum(len(v) for v in final_hard.values())} -> {output_hard_file}")
        logger.info(f"    - Easy (Failed) Items: {sum(len(v) for v in final_easy.values())} -> {output_easy_file}")

if __name__ == "__main__":
    refiner = BenchmarkRefiner()
    refiner.run_pipeline()