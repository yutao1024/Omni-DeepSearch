import os
import json
import re
import logging
from tool_text_search import search as mm_text_search
from llm_provider import get_llm_provider

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("VisualFilter")

class VisualNecessityTester:
    def __init__(self, config, prompts):
        self.config = config
        agentic_cfg = prompts.get('agentic_search', {})
        self.system_prompt = agentic_cfg.get('system', '')

        provider_name = self.config['llm'].get('filter_provider', 'gpt')
        self.llm = get_llm_provider(self.config['llm'], provider_name)

    def run_blind_search(self, item):
        question = item['challenge']['challenges'][0]['question']
        identity_hint = item.get('video_info', {}).get('identity', 'Unknown Subject')
        user_p = (
            f"SUBJECT IDENTITY: {identity_hint}\n"
            f"QUESTION: {question}\n\n"
            f"INSTRUCTION: You can ONLY use text_search to query textual sources. "
            f"Based on the identity '{identity_hint}', search if any text explicitly mentions "
            f"the visual answer. If nothing is found, you MUST conclude that you cannot answer."
        )
        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_p
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_p},
            {"role": "assistant", "content": response}
        ]
        MAX_TURNS = 5
        current_res = response
        for turn in range(MAX_TURNS):
            if not current_res or "Search failed" in current_res:
                break
            if "<answer>" in current_res:
                ans_match = re.search(r'<answer>(.*?)</answer>', current_res, re.S)
                return ans_match.group(1).strip() if ans_match else current_res
            if "<tool_call>" in current_res:
                try:
                    tool_json = json.loads(re.search(r'\{.*\}', current_res, re.DOTALL).group())
                    args = tool_json.get("arguments", tool_json)
                    queries = args.get("query_list", [identity_hint])
                    logger.info(f"      [Action] Text Search: {queries}")
                    search_res = mm_text_search(queries, self.config, question, current_res)
                    messages.append({"role": "user", "content": f"<tool_response>\n{search_res['result']}\n</tool_response>"})
                    current_res = self.llm.generate(messages=messages)
                    messages.append({"role": "assistant", "content": current_res})
                except Exception as e:
                    logger.error(f"      [Tool Parse Error]: {e}")
                    break
            else:
                return current_res
        return "Search inconclusive."

    def check_leakage(self, prediction, ground_truth):
        if len(prediction) < 5 or "inconclusive" in prediction.lower() or "cannot answer" in prediction.lower():
            return False
        sys_p = "You are a high-precision semantic auditor."
        user_p = f"Does the Prediction accurately describe the Ground Truth visual detail?\n[GT]: {ground_truth}\n[Pred]: {prediction}\nAnswer 'YES' or 'NO'."
        res = self.llm.generate(system_prompt=sys_p, user_prompt=user_p)
        return "YES" in res.upper()

    def process_json_file(self, input_path):
        if not os.path.exists(input_path):
            logger.error(f"File not found: {input_path}")
            return
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"========== Starting Visual Necessity Test ==========")
        verified_data = {}
        total_kept = 0
        total_leaked = 0
        for category, items in data.items():
            kept_items = []
            logger.info(f"--- Testing Category: {category} ({len(items)} items) ---")
            for idx, item in enumerate(items):
                sample_id = item.get('sample_id', f'item_{idx}')
                logger.info(f"[{idx+1}/{len(items)}] Testing: {sample_id}")
                gt_obj = item['challenge']['challenges'][0]['ground_truth_answer']
                if isinstance(gt_obj, dict):
                    gt_visual_desc = gt_obj.get('visual_verification_answer') or gt_obj.get('visual_action_description', str(gt_obj))
                else:
                    gt_visual_desc = str(gt_obj)
                prediction = self.run_blind_search(item)
                if self.check_leakage(prediction, gt_visual_desc):
                    logger.warning(f"    => [REJECT] Information leaked!")
                    total_leaked += 1
                else:
                    logger.info(f"    => [KEEP] Secure visual detail.")
                    kept_items.append(item)
                    total_kept += 1
            verified_data[category] = kept_items
        output_path = input_path.replace(".json", "_VISUAL_VERIFIED.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(verified_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Done! Kept: {total_kept}, Rejected: {total_leaked}")
        logger.info(f"Saved verified items to: {output_path}")
        return output_path