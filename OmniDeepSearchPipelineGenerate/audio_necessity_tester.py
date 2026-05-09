import os
import json
import re
import yaml
import base64
import mimetypes
import logging
import asyncio

from llm_provider import get_llm_provider
from tool_video_search import func_video_search_by_text_query
from tool_text_search import search as mm_text_search
from tool_image_search_text_query import func_image_search_by_text_query

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("AudioNecessityAgent")

class AudioNecessityTester:
    def __init__(self, config_path="config/config.yaml", prompt_path="config/prompts.yaml", max_concurrent_tasks=10):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, config_path), 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(os.path.join(base_dir, prompt_path), 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        
        self.llm = get_llm_provider(self.config['llm'], self.config['llm']['filter_provider'])
        self.initial_analysis = "" 
        
        self.semaphore = None
        self.save_lock = None
        self.max_concurrent_tasks = max_concurrent_tasks
        
    def _read_file_b64_sync(self, media_path):
        """同步读取文件并进行base64编码的辅助方法"""
        with open(media_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    async def _encode_media_to_message(self, media_path):
        if not os.path.exists(media_path):
            return None
        ext = media_path.lower()

        b64_data = await asyncio.to_thread(self._read_file_b64_sync, media_path)
            
        if ext.endswith((".wav", ".mp3", ".ogg")):
            return {"type": "input_audio", "input_audio": {"data": b64_data, "format": "wav"}}
        elif ext.endswith((".jpg", ".jpeg", ".png")):
            mime_type, _ = mimetypes.guess_type(media_path)
            mime_type = mime_type or 'image/jpeg'
            return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}}
        return None

    def _get_task_specific_system_prompt(self, raw_system_prompt, is_video_task):
        search_prompt = self.prompts.get('agentic_search_first_node', {}).get('system', None)
        
        if not search_prompt:
            search_prompt = raw_system_prompt
        
        return search_prompt.strip()

    async def _guess_audio_subject(self, question):
        sys_p = (
            "You are an expert audio detective. Your task is to identify the main subject of an audio clip based on the provided text question. "
            "Crucially, if the clues provided in the question are insufficient, ambiguous, or do not point to a single high-confidence entity, "
            "you should admit that you cannot identify the subject. "
            "DO NOT guess, do not hallucinate."
        )
        user_p = (
            f"Question: {question}\n\n"
            "Task: Identify the main subject (person, object, animal, or phenomenon) of the audio.\n"
            "Strict Rules:\n"
            "1. If you are certain, wrap the entity name in <subject>...</subject> tags.\n"
            "2. If the clues are insufficient to identify a specific subject with high confidence, "
            "   you MUST output ONLY the word: <subject>UNKNOWN</subject>.\n"
            "3. Never guess based on limited information. If in doubt, output UNKNOWN."
        )
        
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": [{"type": "text", "text": user_p}]}
        ]
        
        response = await self.llm.agenerate(messages=messages)
        if not response:
            return ""
            
        match = re.search(r'<subject>(.*?)</subject>', response, re.S)
        if match:
            return match.group(1).strip()
        return response.strip()
        
    async def _is_entity_match(self, guess, ground_truth):
        if not guess or not ground_truth: return False
        sys_p = "You are an evaluator. Determine if the two entities refer to the exact same person, object, or concept. Answer only YES or NO."
        user_p = f"Entity 1 (Ground Truth): {ground_truth}\nEntity 2 (Model Guess): {guess}\nDo they essentially match?"
        
        res = await self.llm.agenerate(system_prompt=sys_p, user_prompt=user_p)
        if not res: return False
        return "YES" in res.upper()

    async def _guess_all_subjects(self, question, num_subjects):

        sys_p = (
            "You are a master puzzle solver. You are given a text-based search challenge that refers to multiple audio clips you cannot hear.\n"
            f"Your goal is to identify the {num_subjects} distinct subjects (people, animals, objects, or instruments) featured in these clips.\n\n"
            "STRICT RULES:\n"
            "1.If the clues are ambiguous or you cannot identify a specific subject ,you should output <subject>UNKNOWN</subject> for that slot.\n"
            "2. DO NOT hallucinate. Do not 'force' a fit if the evidence is weak.\n"
            "3. If you have any doubt about a subject, it is better to label it as UNKNOWN than to guess."
        )

        user_p = (
            f"Question: {question}\n\n"
            f"Task: Identify the {num_subjects} primary subjects of the audio.\n"
            "For each subject, wrap it in <subject>...</subject> tags.\n"
            "If you cannot identify the subject based on the clues, wrap 'UNKNOWN' in the tags instead.\n"
            "Example for 2 subjects where only the first is known: <subject>Entity 1</subject><subject>UNKNOWN</subject>"
        )
        
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": [{"type": "text", "text": user_p}]}
        ]
        
        response = await self.llm.agenerate(messages=messages)
        if not response: return []
            
        matches = re.findall(r'<subject>(.*?)</subject>', response, re.S)
        return [m.strip() for m in matches]
    async def _verify_all_subjects_match(self, guessed_list, true_list):

        if not guessed_list or not true_list: return False
        
        sys_p = "Determine if the list of 'Guessed Entities' covers all the 'True Entities'. Answer YES or NO."
        user_p = f"True Entities: {', '.join(true_list)}\nGuessed Entities: {', '.join(guessed_list)}\nAre all true entities correctly identified?"
        
        res = await self.llm.agenerate(system_prompt=sys_p, user_prompt=user_p)
        return "YES" in res.upper()



    async def run_blind_search(self, question, is_video_task=False, media_paths=None):
        raw_sys_p = ""  # 不再需要，但签名保留兼容性
        sys_p = self._get_task_specific_system_prompt(raw_sys_p, is_video_task)
        
        text_prompt = question  # 直接传 search_q

        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": [{"type": "text", "text": text_prompt}]}
        ]

        MAX_TURNS = 8
        if is_video_task:
            MAX_TURNS = 5

        search_steps = []

        for turn in range(MAX_TURNS):
            is_last_turn = (turn == MAX_TURNS - 1)
            logger.info(f"      [Turn {turn+1}/{MAX_TURNS}] Multimodal Agent thinking...")

            response = await self.llm.agenerate(messages=messages)
            
            if not response:
                logger.error("LLM returned empty response.")
                break
            
            think_match = re.search(r'<think>(.*?)</think>', response, re.S)
            current_thought = think_match.group(1).strip() if think_match else "[No thinking provided]"

            step_log = {
                "turn": turn + 1,
                "thought": current_thought,
                "tool_name": None,
                "search_queries": None,
                "tool_response_summary": None
            }

            if "<answer>" in response:
                ans_match = re.search(r'<answer>(.*?)</answer>', response, re.S)
                final_ans = ans_match.group(1).strip() if ans_match else response
                
                step_log["action"] = "Final Answer"
                search_steps.append(step_log)
                return final_ans, search_steps

            if "<tool_call>" in response:
                try:
                    tool_json_str = self._extract_tool_json(response)
                    if not tool_json_str:
                        if is_last_turn: 
                            final_ans = self._clean_final_answer(response)
                            step_log["action"] = "Forced Final Answer (Invalid JSON)"
                            search_steps.append(step_log)
                            return final_ans, search_steps
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": [{"type": "text", "text": "Invalid JSON in <tool_call>."}]})
                        
                        step_log["error"] = "Invalid JSON"
                        search_steps.append(step_log)
                        continue 

                    tool_data = json.loads(tool_json_str)

                    queries = (
                        tool_data.get("query_list") or 
                        tool_data.get("queries") or 
                        tool_data.get("arguments", {}).get("query_list") or 
                        tool_data.get("arguments", {}).get("queries") or 
                        tool_data.get("parameters", {}).get("query_list") or 
                        tool_data.get("parameters", {}).get("queries") or    
                        tool_data.get("query") or
                        []
                    )

                    if isinstance(queries, dict):
                        queries = queries.get("query_list") or queries.get("queries") or []
                    if isinstance(queries, str): 
                        queries = [queries]

                    if not queries:
                        potential_keys = [k for k in tool_data.keys() if k not in ["name", "tool_name", "arguments"]]
                        if potential_keys:
                            val = tool_data[potential_keys[0]]
                            queries = [val] if isinstance(val, str) else val
                    if isinstance(queries, str): queries = [queries]
                    
                    tool_name = tool_data.get("name") 
                    
                    step_log["tool_name"] = tool_name
                    step_log["search_queries"] = queries

                    messages.append({"role": "assistant", "content": response})
                    next_user_content = []
                    res_text = ""

                    if tool_name == "text_search":
                        res_dict = await mm_text_search(
                            query_list=queries, config=self.config, 
                            original_question=question, current_thought=current_thought, engine="serper"
                        )
                        res_text = res_dict.get("result", "No information retrieved.")
                        next_user_content.append({"type": "text", "text": f"<tool_response>\n{res_text}\n</tool_response>"})
                    
                    elif tool_name == "video_search":
                        v_paths, res_dict = await func_video_search_by_text_query(
                            query_list=queries, config=self.config,
                            original_question=question, current_thought=current_thought
                        )
                        res_text = res_dict.get("result", "No video retrieved.")
                        next_user_content.append({"type": "text", "text": f"<tool_response>\n{res_text}\n</tool_response>"})
                        for vp in v_paths:
                            v_msg = await self._encode_media_to_message(vp)
                            if v_msg: next_user_content.append(v_msg)

                    elif tool_name == "image_search_by_text_query":
                        img_paths, res_dict = await func_image_search_by_text_query(
                            data_id="agent_blind", 
                            query_list=queries, 
                            engine="serper", 
                            search_api_key=self.config.get('serper_api_key')
                        )
                        res_text = res_dict.get("result", "No images retrieved.")
                        next_user_content.append({"type": "text", "text": f"<tool_response>\n{res_text}\n</tool_response>"})
                        for img_path in img_paths:
                            img_msg = await self._encode_media_to_message(img_path[0] if isinstance(img_path, list) else img_path)
                            if img_msg: next_user_content.append(img_msg)
                                
                    else:
                        res_text = "Unknown tool."
                        next_user_content.append({"type": "text", "text": f"<tool_response>Unknown tool.</tool_response>"})

                    step_log["tool_response_summary"] = res_text[:1000] + ("..." if len(res_text) > 1000 else "")
                    search_steps.append(step_log)

                    if is_last_turn:
                        messages.append({"role": "user", "content": [{"type": "text", "text": "Final chance. Provide <answer>."}]})
                        final_response = await self.llm.agenerate(messages=messages)
                        final_ans = self._clean_final_answer(final_response)
                        
                        search_steps.append({
                            "turn": "Final Force",
                            "thought": "Forced to answer due to turn limit.",
                            "final_answer": final_ans
                        })
                        return final_ans, search_steps
                    else:
                        messages.append({"role": "user", "content": next_user_content})
                        continue

                except Exception as e:
                    logger.error(f"Error in blind search loop: {e}")
                    step_log["error"] = str(e)
                    search_steps.append(step_log)
                    break
 
        return response, search_steps

    def _extract_tool_json(self, response):
        if not response: return None
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.S)
        content = tool_match.group(1).strip() if tool_match else response
        content = re.sub(r'```json\s*|```', '', content)
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx+1]
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    while json_str and json_str.count('}') > json_str.count('{'):
                        json_str = json_str[:json_str.rfind('}')].strip()
                        if json_str.endswith('}'): break
                    return json_str if json_str else None
        except: pass
        matches = re.findall(r'\{.*\}', content, re.S)
        return matches[-1] if matches else None

    def _clean_final_answer(self, text):
        if not text: return ""
        ans_match = re.search(r'<answer>(.*?)</answer>', text, re.S)
        if ans_match: return ans_match.group(1).strip()
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
        cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.S)
        cleaned = re.sub(r'<.*?>', '', cleaned)
        return cleaned.strip()

    def _save_progress_sync(self, out_nec, out_ind, out_log, necessary_data, independent_data, analysis_log):
        with open(out_nec, 'w', encoding='utf-8') as f:
            json.dump(necessary_data, f, indent=2, ensure_ascii=False)
        with open(out_ind, 'w', encoding='utf-8') as f:
            json.dump(independent_data, f, indent=2, ensure_ascii=False)
        with open(out_log, 'w', encoding='utf-8') as f:
            json.dump(analysis_log, f, indent=2, ensure_ascii=False)

    async def _process_single_item(self, cat, item, necessary_data, independent_data, analysis_log, out_nec, out_ind, out_log):
        async with self.semaphore:
            try:
                q = item['challenge']['challenges'][0]['question']
                gt_answer = item['challenge']['challenges'][0]['ground_truth_answer']
                is_video_task = "visual_ground_truth_dir" in item or (isinstance(gt_answer, dict) and "visual_action_description" in gt_answer)
                
                kg_info = item.get('kg_info', {})
                node_sequence = kg_info.get('node_sequence', [])

                log_entry = {
                    "sample_id": item['sample_id'],
                    "question": q,
                    "search_steps": [],
                    "filter_status": ""
                }

                if cat == "INTERACTION":
                    true_subjects = [c.get('entity', 'Unknown') for c in item.get('video_info', {}).get('clips', [])]
                    first_node = node_sequence[1] if len(node_sequence) > 1 else str(gt_answer)
                    
                    log_entry.update({
                        "true_subjects": true_subjects,
                        "true_first_node": first_node
                    })
                    logger.info(f"    [Start Interaction] Sample: {item['sample_id']} | Targets: {true_subjects}")

                    guessed_subjects = await self._guess_all_subjects(q, len(true_subjects))
                    log_entry["guessed_subjects"] = guessed_subjects
                    
                    if await self._verify_all_subjects_match(guessed_subjects, true_subjects):
                        logger.warning(f"    [FILTER 1] All subjects {true_subjects} leaked via text -> Independent")
                        log_entry["filter_status"] = "Filtered by Filter 1 (All Subjects Guessed)"
                        async with self.save_lock:
                            independent_data[cat].append(item)
                            analysis_log.append(log_entry)
                            await asyncio.to_thread(self._save_progress_sync, out_nec, out_ind, out_log, necessary_data, independent_data, analysis_log)
                        return

                    search_q = (
                    f"This is a Multi-Audio Interaction challenge. You CANNOT hear the clips.\n"
                    f"TASK: Identify the intersection (Bridge Entity) of the subjects mentioned in the question, "
                    f"then use search tools to find the VERY FIRST logical jump/node after that bridge.\n"
                    f"Identify this first jump node and wrap it in <answer>...</answer>. "
                    f"IMPORTANT: If you cannot find a clear, logical result through search, or if the information is too ambiguous to confirm, you should abandon the task and state that you cannot complete it. "
                    f"Question: {q}"
                    )
                    
                    blind_node_ans, steps = await self.run_blind_search(search_q, is_video_task=is_video_task)
                    log_entry["guessed_first_node"] = blind_node_ans
                    log_entry["search_steps"] = steps

                    if await self._is_entity_match(blind_node_ans, first_node):
                        logger.warning(f"    [FILTER 2] Logical jump '{first_node}' found via search -> Independent")
                        log_entry["filter_status"] = "Filtered by Filter 2 (Bridge Jump Node Found)"
                        async with self.save_lock:
                            independent_data[cat].append(item)
                            analysis_log.append(log_entry)
                            await asyncio.to_thread(self._save_progress_sync, out_nec, out_ind, out_log, necessary_data, independent_data, analysis_log)
                        return

                else:
                    audio_subject = kg_info.get('start_entity', "")
                    if not audio_subject:
                        audio_subject = item.get('video_info', {}).get('identity', "")
                    
                    first_node = node_sequence[1] if len(node_sequence) > 1 else gt_answer
                    
                    log_entry.update({
                        "true_subject": audio_subject,
                        "true_first_node": first_node
                    })

                    guessed_subject = await self._guess_audio_subject(q)
                    log_entry["guessed_subject"] = guessed_subject

                    if await self._is_entity_match(guessed_subject, audio_subject):
                        logger.warning(f"    [RESULT] Single Subject '{audio_subject}' guessed -> Independent")
                        log_entry["filter_status"] = "Filtered by Filter 1 (Subject Guessed)"
                        async with self.save_lock:
                            independent_data[cat].append(item)
                            analysis_log.append(log_entry)
                            await asyncio.to_thread(self._save_progress_sync, out_nec, out_ind, out_log, necessary_data, independent_data, analysis_log)
                        return

                    search_q = (
                        f"Do NOT try to answer the final question. Instead, use your search tools to identify the VERY FIRST intermediate entity, concept, or person derived from the audio subject mentioned in this question. "
                        f"Identify this first logical node and wrap it in <answer>...</answer>. "
                        f"IMPORTANT: If you cannot find a clear, logical result through search, or if the information is too ambiguous to confirm, you should abandon the task and state that you cannot complete it. "
                        f"The original question is: {q}"
                    )
                    blind_node_ans, steps = await self.run_blind_search(search_q, is_video_task=is_video_task)
                    log_entry["guessed_first_node"] = blind_node_ans
                    log_entry["search_steps"] = steps

                    if await self._is_entity_match(blind_node_ans, first_node):
                        log_entry["filter_status"] = "Filtered by Filter 2"
                        async with self.save_lock:
                            independent_data[cat].append(item)
                            analysis_log.append(log_entry)
                            await asyncio.to_thread(self._save_progress_sync, out_nec, out_ind, out_log, necessary_data, independent_data, analysis_log)
                        return

                logger.info(f"    [RESULT] Sample {item['sample_id']} Audio-Necessary (Verified)")
                log_entry["filter_status"] = "Passed Both Filters"
                async with self.save_lock:
                    necessary_data[cat].append(item)
                    analysis_log.append(log_entry)
                    await asyncio.to_thread(self._save_progress_sync, out_nec, out_ind, out_log, necessary_data, independent_data, analysis_log)

            except Exception as e:
                logger.error(f"Error processing {item.get('sample_id')}: {e}", exc_info=True)

    async def process_json_file(self, file_path):
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.save_lock = asyncio.Lock()

        if not os.path.exists(file_path):
            logger.warning(f"File not found for necessity testing: {file_path}")
            return False

        logger.info(f">>> Testing Audio Necessity for: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        necessary_data = {cat: [] for cat in data.keys()}
        independent_data = {cat: [] for cat in data.keys()}
        analysis_log = []
        
        base_path = os.path.splitext(file_path)[0]
        out_nec = f"{base_path}_AUDIO_NECESSARY.json"
        out_ind = f"{base_path}_AUDIO_INDEPENDENT.json"
        out_log = f"{base_path}_BLIND_GUESS_LOG.json"

        tasks = []
        try:
            for cat, items in data.items():
                logger.info(f"[*] Queueing Category: {cat}")
                for item in items:
                    task = asyncio.create_task(
                        self._process_single_item(
                            cat, item, necessary_data, independent_data, analysis_log, out_nec, out_ind, out_log
                        )
                    )
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)

        except KeyboardInterrupt:
            logger.warning(">>> Process manually interrupted! Saving current progress...")
        except Exception as e:
            logger.error(f">>> Fatal error during processing: {e}")
            raise e


        self._save_progress_sync(out_nec, out_ind, out_log, necessary_data, independent_data, analysis_log)
        return True

