import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('all_proxy', None)
import sys
import json
import yaml
import logging
import asyncio
import re
from datetime import datetime
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_agent import AudioDeepSearchAgent
from inference_evaluator import BenchmarkEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("InferencePipeline")

class InferencePipeline:
    def __init__(self, workspace_root, output_dir=None):
        self.workspace_root = os.path.abspath(workspace_root)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "config.yaml")
        prompt_path = os.path.join(os.path.dirname(__file__), "inference_prompts.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        self.agent = AudioDeepSearchAgent(self.config, self.prompts)
        self.evaluator = BenchmarkEvaluator(self.config, self.prompts)

        if output_dir and os.path.exists(output_dir):
            self.output_dir = os.path.abspath(output_dir)

        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join(self.workspace_root, f"eval_results_{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)

        self.report_file = os.path.join(self.output_dir, "final_report.json")
        self.file_lock = None 

    def _get_all_datasets(self):
        dataset_files = []
        for root, dirs, files in os.walk(self.workspace_root):
            if "dataset.json" in files:
                dataset_files.append(os.path.join(root, "dataset.json"))
        return dataset_files

    def _get_media_paths(self, item, json_dir):
        raw_paths = []
        if "audio_file" in item:
            raw_paths.append(item["audio_file"])
        if "audio_files" in item:
            raw_paths.extend(item["audio_files"])
        
        valid_paths = []
        for p in raw_paths:
            full_p = p if os.path.isabs(p) else os.path.normpath(os.path.join(json_dir, p))
            if os.path.exists(full_p):
                valid_paths.append(full_p)
            else:
                logger.warning(f"File not found: {full_p}")
        return valid_paths

    def _identify_task(self, item, json_path):
        sample_id = str(item.get("sample_id", ""))
        is_video_task = "TRACING" in sample_id or "/trace/" in json_path.lower()
        
        try:
            challenge_data = item.get("challenge", {}).get("challenges", [{}])[0]
            question = challenge_data.get("question", "")
            gt_answer = challenge_data.get("ground_truth_answer", "")
            if not gt_answer and "ground_truth" in item:
                gt_answer = item["ground_truth"].get("visual_verification_answer", "")
        except Exception:
            question, gt_answer = "", ""

        return question, gt_answer, is_video_task

    def _load_progress(self):

        if os.path.exists(self.report_file):
            try:
                with open(self.report_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                report = data.get("detailed_data", {})
                per_question_results = data.get("per_question_status", [])
                
                processed_ids = {item["sample_id"] for item in per_question_results}

                return report, per_question_results, processed_ids
            except Exception as e:
                logger.error(f"Failed to load progress file: {e}. Starting fresh.")
        return {}, [], set()

    def _get_sort_key(self, category_path):
        parts = category_path.lower().split('/')
        macro = parts[0] if len(parts) > 0 else ""
        micro = parts[1] if len(parts) > 1 else ""
        macro_rank = {"single": 1, "image": 2, "trace": 3, "multi": 4}
        micro_rank_std = {"speech": 1, "music": 2, "bio": 3, "env": 4}
        micro_rank_multi = {"2way": 1, "3way": 2, "4way": 3}
        m_rank = macro_rank.get(macro, 99)
        mi_rank = micro_rank_multi.get(micro, 99) if macro == "multi" else micro_rank_std.get(micro, 99)
        return (m_rank, mi_rank, category_path)

    def _calculate_and_save_stats(self, report, per_question_results):
        sorted_categories = sorted(report.keys(), key=self._get_sort_key)
        final_statistics = {}
        macro_statistics = {}
        total_q, total_c = 0, 0
        
        for cat in sorted_categories:
            val = report[cat]
            acc = (val["correct"] / val["total"]) * 100 if val["total"] > 0 else 0
            final_statistics[cat] = {
                "accuracy": f"{acc:.2f}%",
                "correct": val["correct"],
                "total": val["total"]
            }
            total_q += val["total"]
            total_c += val["correct"]

            macro_cat = cat.split('/')[0].upper()
            if macro_cat not in macro_statistics:
                macro_statistics[macro_cat] = {"correct": 0, "total": 0}
            macro_statistics[macro_cat]["correct"] += val["correct"]
            macro_statistics[macro_cat]["total"] += val["total"]

        macro_summary = {}
        macro_order = ["SINGLE", "IMAGE", "TRACE", "MULTI"]
        ordered_macros = [m for m in macro_order if m in macro_statistics] + \
                         [m for m in macro_statistics if m not in macro_order]
                         
        for m in ordered_macros:
            m_val = macro_statistics[m]
            m_acc = (m_val["correct"] / m_val["total"]) * 100 if m_val["total"] > 0 else 0
            macro_summary[m] = {
                "accuracy": f"{m_acc:.2f}%",
                "correct": m_val["correct"],
                "total": m_val["total"]
            }

        overall_acc = (total_c / total_q) * 100 if total_q > 0 else 0
        summary = {
            "overall_accuracy": f"{overall_acc:.2f}%",
            "total_questions": total_q,
            "total_correct": total_c,
            "macro_category_stats": macro_summary,
            "sub_category_stats": final_statistics,
            "per_question_status": per_question_results,
            "detailed_data": report
        }

        tmp_file = self.report_file + ".tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, self.report_file)

        return total_q, total_c, overall_acc, ordered_macros, macro_summary, final_statistics, sorted_categories

    async def process_sample(self, item, json_path, semaphore, report, per_question_results, processed_ids):
        sample_id = str(item.get('sample_id', 'unknown'))

        if sample_id in processed_ids:
            logger.info(f"⏭️ [Skipped] {sample_id} already processed.")
            return

        async with semaphore:
            json_dir = os.path.dirname(json_path)
            category = os.path.relpath(json_dir, self.workspace_root).replace('\\', '/')
            q, gt, is_video_task = self._identify_task(item, json_path)
            media_paths = self._get_media_paths(item, json_dir)
            
            if not q or not media_paths:
                return

            max_retries = 10
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        logger.warning(f" [Retry {attempt}/{max_retries-1}] {sample_id}")
                    
                    logger.info(f"🚀 [Running] {sample_id} | Sub-Category: {category}")
                    
                    answer, thought = await self.agent.run(question=q, media_paths=media_paths, is_video_task=is_video_task)
                    is_correct, votes = await self.evaluator.evaluate(q, gt, answer)
                    
                    status_icon = "✅" if is_correct else "❌"
                    logger.info(f"      Result: {status_icon} for {sample_id}")

                    res_entry = {
                        "sample_id": sample_id,
                        "sub_category": category,
                        "is_correct": is_correct,
                        "question": q,
                        "gt": gt,
                        "agent_answer": answer,
                        "votes": votes,
                        "thought_history": thought,
                        "retries": attempt
                    }
                    
                    async with self.file_lock:
                        if category not in report:
                            report[category] = {"correct": 0, "total": 0, "samples": []}
                        report[category]["total"] += 1
                        if is_correct: report[category]["correct"] += 1
                        report[category]["samples"].append(res_entry)

                        per_question_results.append({
                            "sample_id": sample_id,
                            "sub_category": category,
                            "status": "Correct" if is_correct else "Wrong",
                            "retries": attempt
                        })
                        processed_ids.add(sample_id) 
                        self._calculate_and_save_stats(report, per_question_results)
                    
                    return 

                except Exception as e:
                    if "429" in str(e):
                        wait_time = (2 ** attempt) + random.random() * 2
                        logger.warning(f"⚠️ Rate limited (429). Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Error on {sample_id}: {e}")
                        await asyncio.sleep(2)

    async def run(self):
        self.file_lock = asyncio.Lock() 
        
        report, per_question_results, processed_ids = self._load_progress()
        
        dataset_files = self._get_all_datasets()
        logger.info(f"Found {len(dataset_files)} sub-datasets.")
        
        semaphore = asyncio.Semaphore(5) 
        tasks = []

        for j_file in dataset_files:
            with open(j_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.get("INTERACTION", [])
                for item in items:
                    tasks.append(self.process_sample(
                        item, j_file, semaphore, report, per_question_results, processed_ids
                    ))

        await asyncio.gather(*tasks)

        logger.info(f"\n" + "="*60)
        logger.info(f"EVALUATION FINISHED")
        
        total_q, total_c, overall_acc, ordered_macros, macro_summary, final_statistics, sorted_categories = self._calculate_and_save_stats(report, per_question_results)

        logger.info(f"OVERALL ACCURACY: {overall_acc:.2f}% ({total_c}/{total_q})")
        logger.info("-" * 30)
        logger.info("CATEGORY BREAKDOWN:")
        
        for macro in ordered_macros:
            m_stat = macro_summary[macro]
            logger.info(f"📁 [{macro}] OVERALL: {m_stat['accuracy']} ({m_stat['correct']}/{m_stat['total']})")
            for cat in sorted_categories:
                if cat.upper().startswith(macro):
                    stat = final_statistics[cat]
                    logger.info(f"    ├─ [{cat}]: {stat['accuracy']} ({stat['correct']}/{stat['total']})")
            logger.info("    └─" + "-"*20)
            
        logger.info("-" * 30)
        logger.info(f"Full JSON report strictly saved at: {self.report_file}")
        logger.info("="*60)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python inference_pipeline.py <workspace_path> [resume_dir]")
        sys.exit(1)

    WORKSPACE = sys.argv[1]
    RESUME_DIR = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(WORKSPACE):
        logger.error(f"Workspace path not found: {WORKSPACE}")
        sys.exit(1)

    pipeline = InferencePipeline(workspace_root=WORKSPACE, output_dir=RESUME_DIR)
    asyncio.run(pipeline.run())