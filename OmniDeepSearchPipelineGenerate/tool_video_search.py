import os
import logging
import copy
import base64
import mimetypes
import cv2
import asyncio
from typing import List, Dict, Tuple
import subprocess
from video_tools import VideoTools
from llm_provider import get_llm_provider

VIDEO_WORKSPACE = "./search_cache" 

logger = logging.getLogger("VideoSearchTool")

async def _is_video_relevant_metadata(original_question: str, current_thought: str, title: str, description: str, config: dict) -> bool:
    try:
        filter_provider_name = config['llm'].get('filter_provider', 'gpt')
        llm = get_llm_provider(config['llm'], filter_provider_name)
        system_prompt = (
            "You are a fast video filtering assistant. "
            "Filter out obvious advertisements or completely useless garbage. "
            "Output 'Yes' if it has ANY potential relevance, otherwise 'No'."
        )
        user_prompt = f"[Question]: {original_question}\n[Reasoning]: {current_thought}\n[Title]: {title}\n[Desc]: {description}\nIs this useful? Yes or No:"
        res = await llm.agenerate(system_prompt=system_prompt, user_prompt=user_prompt, media_files=[])
        return "no" not in (res or "").lower()
    except Exception as e:
        logger.warning(f"Metadata filter failed: {e}")
        return True 

async def _evaluate_16_frames_with_context(conversation_history: list, frame_paths: List[str], config: dict) -> Tuple[bool, str]:

    main_provider_name = config['llm'].get('inference_provider', config['llm'].get('default_provider', 'gemini'))
    try:
        main_llm = get_llm_provider(config['llm'], main_provider_name)
    except Exception as e:
        logger.error(f"Failed to load main LLM for verification: {e}")
        return False, "System Error: LLM Load Failed."

    temp_messages = copy.deepcopy(conversation_history)
    
    user_content = []
    for img_path in frame_paths:
        with open(img_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode('utf-8')
        mime_type, _ = mimetypes.guess_type(img_path)
        user_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:{mime_type or 'image/jpeg'};base64,{b64_data}"}
        })

    user_content.append({
        "type": "text", 
        "text": (
            "[Internal Sandbox Check] Look at these 16 sampled frames from a candidate video. "
            "Based on our previous reasoning, analyze what is shown in these frames versus what we are looking for. "
            "Provide a brief analysis of the visual content. "
            "Finally, you MUST end your response with EXACTLY '[RESULT: YES]' if it is the target video, "
            "or EXACTLY '[RESULT: NO]' if it is irrelevant or incorrect."
        )
    })
    
    temp_messages.append({"role": "user", "content": user_content})
    logger.info(f"      [16-Frame Check] Analyzing 16 frames with Main LLM...")
    
    try:
        res = await main_llm.agenerate(messages=temp_messages)
        res = res if res else "[RESULT: NO] Empty response."
    except Exception as e:
        logger.warning(f"Main LLM 16-frame check failed: {e}")
        res = f"[RESULT: NO] Error during analysis: {e}"

    is_target = "[RESULT: YES]" in res.upper()
    
    feedback = res.replace("[RESULT: YES]", "").replace("[RESULT: NO]", "").strip()
    
    logger.info(f"      [16-Frame Check] Target: {is_target} | Feedback: {feedback[:100]}...")
    return is_target, feedback

def _sync_extract_frames(video_path: str, video_id: str, target_count: int, prefix: str):
    evidence_dir = os.path.join(VIDEO_WORKSPACE, f"evidence_{video_id}")
    os.makedirs(evidence_dir, exist_ok=True)

    existing = sorted([
        f for f in os.listdir(evidence_dir)
        if f.startswith(prefix) and f.endswith(".jpg")
    ])

    if len(existing) >= target_count:
        return [
            os.path.abspath(os.path.join(evidence_dir, f))
            for f in existing[:target_count]
        ]
    evidence_dir = os.path.join(VIDEO_WORKSPACE, f"evidence_{video_id}")
    os.makedirs(evidence_dir, exist_ok=True)

    output_files = []
    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
    except Exception:
        logger.error(f"ERROR!!!!!!!")
        return []

    if duration <= 0:
        return []

    timestamps = [
        (i + 0.5) * duration / target_count
        for i in range(target_count)
    ]

    for i, t in enumerate(timestamps):
        t_sec = int(t)
        img_path = os.path.join(
            evidence_dir,
            f"{prefix}_frame_{i:03d}_{t_sec}s.jpg"
        )

        cmd = [
            "ffmpeg",
            "-hwaccel", "none", 
            "-y",
            "-ss", str(t),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            "-loglevel", "error",
            img_path
        ]

        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(img_path):
                output_files.append(os.path.abspath(img_path))
        except Exception:
            logger.error(f"ERROR!!!!!!!!!")
            continue

    return output_files

def extract_vid(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

async def func_video_search_by_text_query(
    query_list: List[str],
    config: dict,
    original_question: str,
    current_thought: str,
    conversation_history: list,  
    original_media_paths: List[str] = [] 
) -> Tuple[List[str], Dict[str, str]]:
    
    os.makedirs(VIDEO_WORKSPACE, exist_ok=True)
    orig_audio = original_media_paths[0] if original_media_paths else None

    query = query_list[0] if query_list else ""
    if not query:
        return [], {"result": "Error: Empty query list provided."}

    results = await VideoTools.search_videos(query, config_dict=config, max_results=1)
    if not results:
        return [], {"result": f"[Video Search] No results found for '{query}'. Please try completely different keywords."}

    for item in results:
        video_url = item['url']
        meta = await VideoTools.fetch_video_metadata(video_url, config_dict=config)
        video_title = meta.get('full_title', 'Unknown Title')

        if not await _is_video_relevant_metadata(original_question, current_thought, video_title, meta.get('full_description', ''), config):
            continue
        
        vid_id = extract_vid(video_url)
        save_dir = os.path.join(VIDEO_WORKSPACE, "downloads")
        os.makedirs(save_dir, exist_ok=True)

        expected_path = os.path.join(save_dir, f"{vid_id}.mp4")

        if vid_id and os.path.exists(expected_path):
            logger.info(f"[Cache Hit] Video already exists: {vid_id}")
            video_path = expected_path
        else:
            manifest = await VideoTools.download_videos([video_url], save_dir, config_dict=config, height=480)
            if not manifest:
                continue
            vid_id = list(manifest.keys())[0]
            video_path = manifest[vid_id]['path']
            logger.info(f"      [Testing 1 Video] Title: {video_title}")

        frames_16 = await asyncio.to_thread(_sync_extract_frames, video_path, vid_id, target_count=16, prefix="16f")
        if not frames_16: 
            continue
            
        is_target, feedback = await _evaluate_16_frames_with_context(conversation_history, frames_16, config)
        
        if not is_target:
            logger.info(f"      [Validation Failed] Returning feedback to Agent.")
            return [os.path.abspath(orig_audio)] if orig_audio else [], {
                "result": (
                    f"[Video Validation Failed]\n"
                    f"We downloaded the video '{video_title}'.\n"
                    f"Visual Analysis Feedback: {feedback}\n"
                    f"Conclusion: This is NOT the correct video. "
                    f"Please reflect on this feedback, generate a NEW search query, and try again."
                )
            }
        
        logger.info(f"      [Validation Success] Generating 64 dense frames.")
        frames_64 = await asyncio.to_thread(_sync_extract_frames, video_path, vid_id, target_count=64, prefix="64f")
        
        final_evidence_paths = [os.path.abspath(orig_audio)] if orig_audio else []
        final_evidence_paths.extend(frames_64)
        
        return final_evidence_paths, {
            "result": (
                f"[Video Search Succeeded]\n"
                f"Video Verified: '{video_title}'.\n"
                f"Analysis: {feedback}\n"
                f"Action: 64 detailed frames have been successfully extracted and added to your context.\n"
                f"CRITICAL INSTRUCTION: You now have the correct video frames. DO NOT make any further tool calls. "
                f"You MUST immediately output the final <answer> based on these frames."
            )
        }

    return [], {
        "result": f"[Video Search] Searched for '{query}', but all top videos were either ads, irrelevant, or failed to download. Please adjust your search strategy."
    }