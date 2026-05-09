import asyncio
import logging
from llm_provider import get_llm_provider

logger = logging.getLogger("InferenceEvaluator")

class BenchmarkEvaluator:
    def __init__(self, config, prompts):
        self.config = config
        self.prompts = prompts
        self.judges = {
            "gpt_judge": get_llm_provider(self.config['llm'], "gpt"),
            "gemini_judge": get_llm_provider(self.config['llm'], "gemini"),
            "claude_judge": get_llm_provider(self.config['llm'], "claude")
        }

    async def evaluate(self, question: str, ground_truth: str, agent_answer: str):
        if not agent_answer:
            return False, {"error": "No response"}
            
        sys_p = self.prompts.get('inference_judge', {}).get('system', "Evaluate Yes/No.")
        user_p = self.prompts.get('inference_judge', {}).get('user_template', "").format(
            question=question, 
            ground_truth_answer=ground_truth, 
            model_response=agent_answer
        )
        
        judge_names = list(self.judges.keys())
        coroutines = [
            self.judges[name].agenerate(system_prompt=sys_p, user_prompt=user_p) 
            for name in judge_names
        ]
        
        raw_results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        results = {}
        yes_count = 0
        
        for idx, name in enumerate(judge_names):
            res = raw_results[idx]
            if isinstance(res, Exception):
                results[name] = f"Error: {res}"
            else:
                is_yes = res and "YES" in res.upper()
                results[name] = "Yes" if is_yes else "No"
                if is_yes:
                    yes_count += 1

        final_decision = yes_count >= 2

        logger.info(f"      [VOTE] GPT:{results.get('gpt_judge')} | Gemini:{results.get('gemini_judge')} | Claude:{results.get('claude_judge')} -> FINAL: {final_decision}")
        
        return final_decision, results