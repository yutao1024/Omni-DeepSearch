import os
import sys
import yt_dlp
import asyncio
import hashlib
import aiohttp
import json
import cv2
import tempfile
import re
import shutil
import logging
from typing import List, Dict, Any
from async_utils import async_retry, run_in_thread
from key_manager import key_manager

VENV_PATH = sys.prefix 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_env_node():
    node_binary = os.path.join(VENV_PATH, "bin", "node")
    if os.path.exists(node_binary):
        return node_binary
    import shutil
    return shutil.which("node")

logger = logging.getLogger("VideoTools")

class VideoTools:

    @staticmethod
    async def _async_retry_wrapper(func, *args, max_retries=3, base_delay=2.0, **kwargs):
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = base_delay * (2 ** attempt)
                print(f"[*] Task failed, retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait_time)

    @staticmethod
    async def search_videos(query: str, config_dict="config/config.yaml", max_results=5) -> List[Dict[str, Any]]:
        cached = key_manager.check_cache("serper_video", query)
        if cached: return cached

        async def _perform_search():
            url = "https://google.serper.dev/search"
            payload = {
                "q": f"{query} site:youtube.com",
                "num": 10,
                "gl": "us",
                "hl": "en",
                "type": "videos"
            }

            async with key_manager.search_semaphore:
                while True:
                    current_key = await key_manager.get_active_key("serper")
                    headers = {'X-API-KEY': current_key, 'Content-Type': 'application/json'}
                    
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(url, json=payload, headers=headers, timeout=15) as resp:
                                status_code = resp.status
                                try:
                                    data = await resp.json()
                                except:
                                    data = {"raw_text": await resp.text()}

                                if status_code == 200:
                                    results = []
                                    raw_items = data.get("videos", [])
                                    if not raw_items:
                                        raw_items = data.get("organic", [])

                                    for item in raw_items:
                                        if len(results) >= max_results: break
                                        link = item.get("link", "")
                                        title = item.get("title", "")
                                        snippet = item.get("snippet", "")
                                        duration = item.get("duration", "Unknown")

                                        if "youtube.com/watch?v=" in link or "youtu.be/" in link:
                                            vid_id = "unknown"
                                            if "v=" in link:
                                                try: vid_id = link.split("v=")[1].split("&")[0]
                                                except: pass
                                            elif "youtu.be/" in link:
                                                try: vid_id = link.split("youtu.be/")[1].split("?")[0]
                                                except: pass

                                            results.append({
                                                "id": vid_id, "title": title, "url": link,
                                                "description": snippet, "duration": duration
                                            })
                                    
                                    key_manager.update_cache("serper_video", query, results)
                                    return results

                                action = await key_manager.report_api_error("serper", current_key, status_code, data, query)
                                if action == "RETRY_WAIT":
                                    await asyncio.sleep(5)
                                    continue
                                elif action == "ROTATE":
                                    continue
                                else:
                                    break
                    except Exception as e:
                        logger.error(f"Video Search Network Error: {e}")
                        await asyncio.sleep(2)
                        break
            return []

        return await VideoTools._async_retry_wrapper(_perform_search)

    @staticmethod
    async def download_videos(url_list, save_dir, config_dict="config/config.yaml", height=720, logger_obj=None, min_dur=None, max_dur=None, max_concurrent=5):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        node_executable = get_env_node()
        print(f"[*] Executing with Node: {node_executable}")

        semaphore = asyncio.Semaphore(max_concurrent)

        SYSTEM_MAX_DUR = 1800
        effective_max_dur = max_dur if max_dur and max_dur <= SYSTEM_MAX_DUR else SYSTEM_MAX_DUR

        format_str = f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={height}]+bestaudio/best[height<={height}][ext=mp4]/bestvideo+bestaudio/best"

        filter_str = []
        if min_dur:
            filter_str.append(f"duration >= {min_dur}")
        if effective_max_dur:
            filter_str.append(f"duration <= {effective_max_dur}")
        match_filter = yt_dlp.utils.match_filter_func(" & ".join(filter_str)) if filter_str else None
        if config_dict:
            cookies_path = config_dict.get('cookies_path')
            abs_path = os.path.abspath(os.path.join(BASE_DIR, cookies_path))
        
        ydl_opts = {
                'format': format_str,
                'merge_output_format': 'mp4',
                'outtmpl': f'{save_dir}/%(id)s.%(ext)s',
                'quiet': True,          
                'no_warnings': False,
                'ignoreerrors': False,
                'cookiefile': abs_path,
                'match_filter': match_filter,
                'js_runtimes': {
                    'node': {'path': node_executable}
                } if node_executable else None,
                'remote_components': ['ejs:github'],
                'restrictfilenames': True,
                'windowsfilenames': False,
                'verbose': False,
                'retries': 10,
                'fragment_retries': 15,
                'socket_timeout': 30,
                'buffersize': 1024 * 256,
                'http_chunk_size': 10485760,
            }

        def _sync_download_task(url):
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info: 
                    raise Exception(f"Extract info failed for {url}")
                if 'requested_downloads' not in info:
                    return None
                vid_id = info.get("id")
                final_path = None
                for ext in ['.mp4', '.mkv', '.webm', '.flv']:
                    potential = os.path.join(save_dir, f"{vid_id}{ext}")
                    if os.path.exists(potential):
                        final_path = potential
                        break
                if not final_path:
                    return None
                return vid_id, {"path": final_path, "title": info.get("title"), "url": url, "id": vid_id}

        @async_retry(max_retries=3, base_delay=5.0) 
        async def _wrapped_download(url):
            async with semaphore: 
                return await run_in_thread(_sync_download_task, url)

        tasks = [_wrapped_download(url) for url in url_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        downloaded_manifest = {}
        for res in results:
            if isinstance(res, tuple):
                vid_id, meta = res
                downloaded_manifest[vid_id] = meta

            elif res is None:
                logger.info("[视频跳过] filtered or skipped")

            elif isinstance(res, Exception):
                logger.error(f"[视频下载错误] {res}")
        
        return downloaded_manifest

    @staticmethod
    async def extract_full_audio(video_path, audio_output_path):
        cmd = [
            "ffmpeg", "-y", 
            "-i", str(video_path),
            "-threads", "0",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_output_path),
            "-loglevel", "error",
            "-nostats"
        ]
        process = await asyncio.create_subprocess_exec(*cmd)
        await process.wait()

    @staticmethod
    async def process_multimodal_sampling(video_manifest, interval, workspace, need_audio=True, audio_duration=None):
        async def _get_video_duration(path):
            cmd = [
                "ffprobe", "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                str(path)
            ]
            try:
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, _ = await proc.communicate()
                return float(stdout.decode().strip())
            except Exception as e:
                logger.error(f"Failed to get duration for {path}: {e}")
                return 0.0

        async def _process_single_video(video_id, meta):
            if not meta.get('path') or not os.path.exists(meta['path']):
                return []

            samples_root = os.path.join(workspace, "samples")
            audio_root = os.path.join(workspace, "audio_temp")
            os.makedirs(samples_root, exist_ok=True)
            
            full_audio_path = None
            if need_audio:
                os.makedirs(audio_root, exist_ok=True)
                full_audio_path = os.path.join(audio_root, f"{video_id}_full.wav")
                await VideoTools.extract_full_audio(meta['path'], full_audio_path)

            video_sample_dir = os.path.join(samples_root, video_id)
            os.makedirs(video_sample_dir, exist_ok=True)

            video_data = []
            
            duration_sec = await _get_video_duration(meta['path'])
            if duration_sec <= 0:
                return []

            current_time = 0.0
            
            while current_time + interval <= duration_sec:
                frame_time = current_time + (interval / 2.0)
                m, s = int(frame_time // 60), int(frame_time % 60)
                base_name = f"{video_id}_{m:02d}m{s:02d}s"
                img_path = os.path.join(video_sample_dir, f"{base_name}.jpg")
                
                cmd_img = [
                    "ffmpeg", "-y", 
                    "-ss", str(frame_time),
                    "-i", meta['path'],
                    "-frames:v", "1",
                    "-q:v", "2",
                    "-loglevel", "error",
                    img_path
                ]
                
                proc_img = await asyncio.create_subprocess_exec(*cmd_img)
                await proc_img.wait()

                if os.path.exists(img_path):
                    video_data.append({
                        "video_id": video_id, "window_start": current_time,
                        "timestamp_str": f"{m:02d}:{s:02d}",
                        "image_path": os.path.abspath(img_path),
                        "audio_path": None, "meta": meta,
                        "base_name": base_name, "frame_time": frame_time
                    })
                
                current_time += interval

            if need_audio and full_audio_path and os.path.exists(full_audio_path):
                effective_audio_dur = audio_duration if audio_duration and audio_duration > 0 else interval
                for node in video_data:
                    audio_start = max(0, node["frame_time"] - (effective_audio_dur / 2.0))
                    wav_path = os.path.join(video_sample_dir, f"{node['base_name']}.wav")
                    cmd_audio = [
                        "ffmpeg", "-y",
                        "-i", full_audio_path,
                        "-ss", str(audio_start),
                        "-t", str(effective_audio_dur),
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        wav_path,
                        "-loglevel", "error"
                    ]
                    proc_audio = await asyncio.create_subprocess_exec(*cmd_audio)
                    await proc_audio.wait()
                    node["audio_path"] = os.path.abspath(wav_path)
                
                try: os.remove(full_audio_path)
                except: pass
            
            return video_data

        all_tasks = [_process_single_video(vid_id, meta) for vid_id, meta in video_manifest.items()]
        results = await asyncio.gather(*all_tasks)
        
        flat_results = []
        for r in results: flat_results.extend(r)
        return flat_results

    @staticmethod
    def _clean_description(raw_desc):
        if not raw_desc: return "No description."

        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', raw_desc)
        text = re.sub(r'www\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '', text)

        cleaned_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line: continue

            if re.search(r'\d{1,2}:\d{2}', line):
                cleaned_lines.append(f"[Chapter] {line}")
                continue

            cleaned_lines.append(line)

        final = "\n".join(cleaned_lines)
        return final if final else "No relevant description."

    @staticmethod
    def _process_vtt_file(vtt_path):
        cleaned_lines = []
        last_line = ""

        try:
            with open(vtt_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line == "WEBVTT": continue
                    if "-->" in line: continue
                    if line.isdigit(): continue
                    if line.startswith(("Style:", "NOTE", "Kind:", "Language:")): continue

                    line = re.sub(r'<[^>]+>', '', line)
                    line = re.sub(r'(align|position|line):[^ ]+', '', line)
                    line = re.sub(r'\[Music\]|\[Applause\]|\[Sound effect\]', '', line, flags=re.IGNORECASE)

                    line = re.sub(r'\s+', ' ', line).strip()
                    if not line: continue

                    if line in last_line:
                        continue
                    if last_line and last_line in line:
                        cleaned_lines.pop()
                        cleaned_lines.append(line)
                        last_line = line
                        continue

                    if line != last_line:
                        cleaned_lines.append(line)
                        last_line = line

            return " ".join(cleaned_lines)
        except Exception:
            return None
            
    @staticmethod
    async def fetch_video_metadata(url, config_dict="config/config.yaml"):
        def _sync_fetch():
            if config_dict:
                cookies_path = config_dict.get('cookies_path')
                abs_path = os.path.abspath(os.path.join(BASE_DIR, cookies_path))

            node_executable = get_env_node()
            print(f"[*] Executing with Node: {node_executable}")

            ydl_opts = {
                'skip_download': True, 'quiet': True, 'no_warnings': False,
                'cookiefile': abs_path,
                'extract_flat': True,
                'js_runtimes': {
                    'node': {'path': node_executable}
                } if node_executable else None,
                'remote_components': ['ejs:github']
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    return {
                        "full_title": info.get('title', 'Unknown'),
                        "full_description": VideoTools._clean_description(info.get('description', '')),
                        "uploader": info.get('uploader') or info.get('channel', 'Unknown'),
                        "upload_date": info.get('upload_date'),
                        "duration": info.get('duration'),
                        "view_count": info.get('view_count', 0)
                    }
            except Exception as e:
                print(f"[Metadata Error] {e}")
                return {"full_title": "Unknown", "full_description": "No description."}

        return await asyncio.to_thread(_sync_fetch)

    @staticmethod
    def uniform_select_slices(slices, max_count):
        total = len(slices)

        if max_count <= 1:
            return slices[:1]

        if total <= max_count:
            return slices
        indices = [int(i * (total - 1) / (max_count - 1)) for i in range(max_count)]
        selected = [slices[i] for i in indices]

        selected.sort(key=lambda x: x['window_start'])
        return selected
    
    async def _get_channel_stats(self, url):
        def _sync_fetch():
            cookie_path = os.path.join(BASE_DIR, "config/cookies.txt")
            node_executable = get_env_node()
            print(f"[*] Executing with Node: {node_executable}")
            ydl_opts = {
                'quiet': True, 'no_warnings': False, 'skip_download': True,
                 'cookiefile': cookie_path,
                'js_runtimes': {
                    'node': {'path': node_executable}
                } if node_executable else None,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
                
        try:
            info = await run_in_thread(_sync_fetch) 
            return {
                'views': info.get('view_count', 0),
                'subs': info.get('channel_follower_count') or info.get('subscriber_count', 0),
                'duration': info.get('duration', 0) 
            }
        except Exception as e:
            return {'views': 0, 'subs': 0, 'duration': 0}