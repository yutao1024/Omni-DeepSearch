import os
import sys
import json
import re
import base64
import mimetypes
import logging
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from llm_provider import get_llm_provider
from tool_text_search import search as mm_text_search
from tool_image_search_text_query import func_image_search_by_text_query
from tool_video_search import func_video_search_by_text_query

logger = logging.getLogger("InferenceAgent")

class AudioDeepSearchAgent:
    def __init__(self, config, prompts):
        self.config = config
        self.prompts = prompts
        provider_name = self.config['llm'].get('inference_provider', self.config['llm']['default_provider'])
        self.llm = get_llm_provider(self.config['llm'], provider_name)



        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join("logs/trajectories", f"session_run_{timestamp}.json")
        os.makedirs("logs/trajectories", exist_ok=True)
    def _encode_media_to_message(self, media_path):
        if not os.path.exists(media_path):
            logger.warning(f"Media file not found: {media_path}")
            return None
            
        ext = media_path.lower()
        with open(media_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode('utf-8').replace('\n', '').replace('\r', '')
            
        if ext.endswith((".wav", ".mp3", ".ogg")):
            return {
                "type": "input_audio", 
                "input_audio": {
                    "data": b64_data, 
                    "format": "wav" if not ext.endswith(".mp3") else "mp3"
                }
            }

        elif ext.endswith((".jpg", ".jpeg", ".png")):
            mime_type, _ = mimetypes.guess_type(media_path)
            mime_type = mime_type or 'image/jpeg'
            return {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_data}"
                }
            }
            
        return None
        
    def _get_task_specific_system_prompt(self, raw_system_prompt, is_video_task):
        prompt = raw_system_prompt.strip()
        
        if is_video_task:
            video_directive = (
                "\n\n[CRITICAL DIRECTIVE: VIDEO TRACING TASK]\n"
                "This is a video-based investigative task. You are provided with several tools, "
                "but for THIS specific task, you are REQUIRED to use 'video_search' only. "
                "Do NOT use 'text_search' or 'image_search_by_text_query'. "
                "Your goal is to find the visual source of the provided audio using 'video_search'."
            )
            prompt += video_directive
            
        return prompt

    async def run(self, question: str, media_paths: list, is_video_task) -> tuple[str, list]:
        raw_sys_p = self.prompts.get('inference_agent', {}).get('system', "You are a multimodal AI assistant.")
        sys_p = self._get_task_specific_system_prompt(raw_sys_p, is_video_task)
        user_content = []
        for path in media_paths:
            media_msg = self._encode_media_to_message(path)
            if media_msg:
                user_content.append(media_msg)
        
        user_content.append({
        "type": "text", 
        "text": (
            "COMMAND: Analyze the provided audio media first. "
            "In your FIRST response, you are REQUIRED to describe the audio content in detail "
            "inside the <think> tag (describe each track if there are multiple). "
            "Identify the 'intersection' if multiple tracks exist, or the 'core clue' if only one exists. "
            "Formulate your search queries based on these audio leads.\n\n"
            f"Question: {question}"
        )
    })
        
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_content}
        ]

        thought_history = []

        MAX_TURNS = 10 if not is_video_task else 3
            
        for turn in range(MAX_TURNS):
            is_last_turn = (turn == MAX_TURNS - 1)
            logger.info(f"      [Turn {turn+1}/{MAX_TURNS}] Agent is thinking/acting...")
            
            response = await self.llm.agenerate(messages=messages) 

            retry_count = 0
            while retry_count <= 3:  
                has_tool = "<tool_call>" in response
                has_answer = "<answer>" in response
                
                if not has_tool and not has_answer:
                    logger.warning(f"      [Fixing] Agent stopped after thinking. Step {turn}, Retry {retry_count+1}")
                    
                    messages.append({"role": "assistant", "content": response})

                    if is_video_task:
                        retry_instruction = (
                            "CONTINUE: You have only provided the <think> block. "
                            "This is a VIDEO TRACING task. You MUST output a <tool_call> block "
                            "specifically using the 'video_search' tool IMMEDIATELY to proceed."
                        )
                    else:
                        retry_instruction = (
                            "CONTINUE: You have only provided the <think> block. "
                            "Now, you MUST output the <tool_call> or <answer> block IMMEDIATELY to proceed."
                        )
                    
                    messages.append({
                        "role": "user", 
                        "content": retry_instruction
                    })
                    
                    new_part = await self.llm.agenerate(messages=messages)

                    response = new_part 
                    retry_count += 1
                else:
                    break

            if not response:
                logger.error("      [Error] LLM returned empty response or was blocked by content filter.")
                thought_history.append({
                    "turn": turn + 1,
                    "thought": "LLM response was empty or blocked by safety filter.",
                    "tool": None,
                    "final_answer": "BLOCKED"
                })
                return "", thought_history

            
            turn_record = {
                "turn": turn + 1,
                "thought": "",
                "tool": None,        
                "tool_output": None, 
                "final_answer": None
            }
            
            think_match = re.search(r'<think>(.*?)</think>', response, re.S)
            if think_match:
                thought_process = think_match.group(1).strip() 
                turn_record["thought"] = thought_process
                logger.info(f"\n{'='*50}\n[💡 Agent Thought Process]:\n{thought_process}\n{'='*50}")
            else:
                thought_process = response.split("<tool_call>")[0].split("<answer>")[0].strip()
                if not thought_process:
                    thought_process = "[Model bypassed think protocol]"
                turn_record["thought"] = thought_process
                logger.info(f"\n{'='*50}\n[💡 Agent Thought (No Tags)]:\n{thought_process[:300]}...\n{'='*50}")

            if "<answer>" in response:
                if turn == 0:
                    logger.warning("      [Intercept] Agent tried to answer early.")
                    messages.append({"role": "assistant", "content": response})
                    
                    if is_video_task:
                        reproach_msg = "You are NOT allowed to answer yet. This is a video tracing task. You MUST use 'video_search' to verify the visual evidence first."
                    else:
                        reproach_msg = "You are NOT allowed to answer in the first turn. You MUST use a search tool."
                    
                    messages.append({"role": "user", "content": reproach_msg})
                    continue 
                ans_match = re.search(r'<answer>(.*?)</answer>', response, re.S)
                ans_text = ans_match.group(1).strip() if ans_match else response
                turn_record["final_answer"] = ans_text
                thought_history.append(turn_record) 
                self._save_trajectory(question, thought_history) 
                return ans_text, thought_history

            if "<tool_call>" in response:
                try:
                    tool_json_str = self._extract_tool_json(response)
                    if not tool_json_str:
                        turn_record["tool"] = {"error": "Invalid tool JSON format"}
                        
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": [{"type": "text", "text": "Invalid JSON format in <tool_call>. Fix it."}]})
                        continue
                    
                    tool_data = json.loads(tool_json_str)


                    tool_name = (
                        tool_data.get("name") or 
                        tool_data.get("tool_name") or 
                        tool_data.get("function", {}).get("name") or
                        "text_search"
                    )

                    params_containers = [
                        tool_data, 
                        tool_data.get("arguments", {}), 
                        tool_data.get("parameters", {}),
                        tool_data.get("function", {}).get("parameters", {})
                    ]

                    queries = []
                    for container in params_containers:
                        if not isinstance(container, dict): continue
                        found_queries = container.get("query_list") or container.get("queries") or container.get("query")
                        if found_queries:
                            queries = found_queries
                            break

                    if isinstance(queries, str):
                        queries = [queries]
                    elif isinstance(queries, dict):
                        queries = list(queries.values())

                    if not queries:
                        for k, v in tool_data.items():
                            if isinstance(v, list) and k not in ["required", "stop"]:
                                queries = v
                                break

                    logger.info(f"      [🔧 Action] Executing {tool_name} for queries: {queries}")
                    
                    turn_record["tool"] = {"name": tool_name, "queries": queries}

                    messages.append({"role": "assistant", "content": response})
                    next_user_content = []

                    if tool_name == "text_search":
                        res_dict = await mm_text_search(  
                            query_list=queries, 
                            config=self.config, 
                            original_question=question,
                            current_thought=thought_process,
                            engine="serper"
                        )
                        res_text = res_dict.get("result", "No information retrieved.")
                        next_user_content.append({"type": "text", "text": f"<tool_response>\n{res_text}\n</tool_response>"})
                        turn_record["tool_output"] = res_text
                    elif tool_name == "video_search":
                        v_media_paths, res_dict = await func_video_search_by_text_query(
                            query_list=queries,
                            config=self.config,
                            original_question=question,
                            current_thought=thought_process,
                            conversation_history=messages,   
                            original_media_paths=media_paths 
                        )
                        res_text = res_dict.get("result", "No video info retrieved.")
                        next_user_content.append({"type": "text", "text": f"<tool_response>\n{res_text}\n</tool_response>"})
                        turn_record["tool_output"] = res_text
                        for vp in v_media_paths:
                            v_msg = self._encode_media_to_message(vp)
                            if v_msg: next_user_content.append(v_msg)
                    elif tool_name == "image_search_by_text_query":
                        img_paths, res_dict = await func_image_search_by_text_query( 
                            data_id="agent_infer", query_list=queries, engine="serper", 
                            search_api_key=self.config.get('serper_api_key')
                        )
                        res_text = res_dict.get("result", "No images retrieved.")
                        next_user_content.append({"type": "text", "text": f"<tool_response>\n{res_text}\n</tool_response>"})
                        turn_record["tool_output"] = res_text
                        for img_p in img_paths:
                            img_msg = self._encode_media_to_message(img_p[0] if isinstance(img_p, list) else img_p)
                            if img_msg: next_user_content.append(img_msg)
                    else:
                        next_user_content.append({"type": "text", "text": f"<tool_response>Unknown tool: {tool_name}</tool_response>"})

                    thought_history.append(turn_record) 
                    self._save_trajectory(question, thought_history) 
                    if is_last_turn:
                        messages.append({"role": "user", "content": next_user_content})
                        
                        logger.info("      [Final Turn] Sending final context without forced prompts.")
                        final_response = await self.llm.agenerate(messages=messages) 
                        
                        final_record = {
                            "turn": "final_turn",
                            "thought": "",
                            "tool": None,
                            "final_answer": None
                        }
                        
                        final_think = re.search(r'<think>(.*?)</think>', final_response, re.S)
                        if final_think:
                            final_record["thought"] = final_think.group(1).strip()
                            logger.info(f"\n{'='*50}\n[💡 Final Agent Thought]:\n{final_record['thought']}\n{'='*50}")
                        ans_match = re.search(r'<answer>(.*?)</answer>', final_response, re.S)
                        if ans_match:
                            ans = ans_match.group(1).strip()
                        else:
                            think_blocks = re.findall(r'<think>(.*?)</think>', final_response, re.S)

                            if think_blocks:
                                last_think = think_blocks[-1].strip()
                            else:
                                last_think = re.sub(r'<.*?>', '', final_response, flags=re.S).strip()

                            MAX_LEN = 500

                            if len(last_think) > MAX_LEN:
                                ans = last_think[:MAX_LEN] + "..."
                            else:
                                ans = last_think
                        final_record["final_answer"] = ans
                        thought_history.append(final_record)
                        self._save_trajectory(question, thought_history)
                        return ans, thought_history
                    else:
                        messages.append({"role": "user", "content": next_user_content})
                        continue

                except Exception as e:
                    logger.error(f"Error processing tool in agent loop: {e}")
                    turn_record["tool"] = {"error": f"Tool execution crashed: {str(e)}"}
                    self._save_trajectory(question, thought_history)
                    thought_history.append(turn_record)
                    break
            else:
                thought_history.append(turn_record)
                self._save_trajectory(question, thought_history)
                return response, thought_history

        return response, thought_history

    def _extract_tool_json(self, response):
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.S)
        content = tool_match.group(1).strip() if tool_match else response
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                return content[start_idx:end_idx+1]
        except: pass
        return None

    def _clean_final_answer(self, text):
        if not text: return ""
        ans_match = re.search(r'<answer>(.*?)</answer>', text, re.S)
        return ans_match.group(1).strip() if ans_match else text

    def _save_trajectory(self, question, trajectory):
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            else:
                all_data = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}

            found = False
            for run in all_data["runs"]:
                if run["question"] == question:
                    run["trajectory"] = trajectory
                    run["update_time"] = time.strftime("%H:%M:%S")
                    found = True
                    break
            
            if not found:
                all_data["runs"].append({
                    "question": question,
                    "update_time": time.strftime("%H:%M:%S"),
                    "trajectory": trajectory
                })

            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")