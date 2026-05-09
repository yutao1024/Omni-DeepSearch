import base64
import asyncio
import aiohttp
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from serpapi import GoogleSearch
from wiki_utils import download_and_resize_image 
from async_utils import async_retry, run_in_thread
from key_manager import key_manager
import logging
logger = logging.getLogger(__name__)


LOG_PATH = "./search_log/serpapi_image_query_log.jsonl"
DOWNLOAD_IMG_DIR = "./search_cache" 
DEFAULT_MAX_IMAGE_PIXELS = 1024 * 1024


def _normalize_data_id(data_id: Any) -> str:
    return str(data_id)

def _normalize_query(query: str) -> str:
    return query.strip().replace(" ", "_")

def _fail_result(msg: str) -> Tuple[List[str], Dict[str, str]]:
    return [], {"result": msg}

def _success_result(titles: List[str]) -> Tuple[List[str], Dict[str, str]]:
    titles = [t for t in titles if t]
    titles_str = "; ".join(titles) if titles else "N/A"
    return [], {
        "result": (
            "[Image Search Succeeded] Relevant image(s) have been successfully retrieved. "
            f"The associated title(s) are: {titles_str}. "
            "The retrieved visual evidence can now be used for downstream multimodal reasoning."
        )
    }

async def append_jsonl_async(log_path: str, obj: Dict[str, Any]) -> None:
    def _sync_append():
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    await run_in_thread(_sync_append)



@async_retry(max_retries=3)
async def get_serper_image_response(query: str, **kwargs) -> Dict[str, Any]:
    cached = key_manager.check_cache("serper_image", query)
    if cached: return cached

    async with key_manager.search_semaphore:
        while True:
            current_key = await key_manager.get_active_key("serper")
            headers = {"X-API-KEY": current_key, "Content-Type": "application/json"}
            url = "https://google.serper.dev/images"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={"q": query}, headers=headers, timeout=15) as resp:
                        status_code = resp.status
                        try:
                            data = await resp.json()
                        except:
                            data = {"raw_text": await resp.text()}

                        if status_code == 200:
                            key_manager.update_cache("serper_image", query, data)
                            return data


                        action = await key_manager.report_api_error("serper", current_key, status_code, data, query)
                        if action == "RETRY_WAIT":
                            await asyncio.sleep(5)
                            continue
                        elif action == "ROTATE":
                            continue
                        else:
                            break
            except Exception as e:
                logger.error(f"Image Search Network Error: {e}")
                await asyncio.sleep(2)
                break
    return {}

async def get_serpapi_image_response(query: str, search_api_key: str) -> List[Dict[str, Any]]:
    """使用线程池执行同步的 SerpApi 请求"""
    def _sync_search():
        params = {
            "engine": "google_images_light",
            "q": query,
            "api_key": search_api_key,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("images_results", [])
    
    return await run_in_thread(_sync_search)


async def _collect_images_from_results(
    data_id: Any,
    query: str,
    retrievals: List[Dict[str, Any]],
    image_keys: Tuple[str, ...],
    topk: int,
) -> List[Dict[str, str]]:

    os.makedirs(DOWNLOAD_IMG_DIR, exist_ok=True)

    norm_data_id = _normalize_data_id(data_id)
    norm_query = _normalize_query(query)
    title_img_list: List[Dict[str, str]] = []

    for idx, retrieval in enumerate(retrievals):
        img_path = os.path.join(DOWNLOAD_IMG_DIR, f"{norm_data_id}_q{norm_query}_{idx}.jpg")

        if os.path.exists(img_path):
            title_img_list.append({
                "cached_title": retrieval.get("title"),
                "cached_images_path": img_path,
            })
            if len(title_img_list) >= topk:
                break
            continue

        saved_path = None
        for key in image_keys:
            image_url = retrieval.get(key)
            if not image_url:
                continue
            try:
                saved_path = await download_and_resize_image(
                    image_url,
                    img_path,
                    max_image_pixels=DEFAULT_MAX_IMAGE_PIXELS,
                )
                if saved_path:
                    break
            except Exception:
                continue

        if saved_path is None:
            continue

        title_img_list.append({
            "cached_title": retrieval.get("title"),
            "cached_images_path": saved_path,
        })
        if len(title_img_list) >= topk:
            break

    return title_img_list


async def parse_serper_image_response(
    data_id: Any,
    search_results: Dict[str, Any],
    query: str,
    topk: int = 3,
) -> List[Dict[str, str]]:
    retrievals = search_results.get("images", [])
    if not retrievals:
        return []

    return await _collect_images_from_results(
        data_id=data_id,
        query=query,
        retrievals=retrievals,
        image_keys=("imageUrl", "thumbnailUrl", "thumbnail"),
        topk=topk,
    )


async def parse_serpapi_image_response(
    data_id: Any,
    search_results: List[Dict[str, Any]],
    query: str,
    topk: int = 3,
) -> List[Dict[str, str]]:
    if not search_results:
        return []

    return await _collect_images_from_results(
        data_id=data_id,
        query=query,
        retrievals=search_results,
        image_keys=("original", "serpapi_thumbnail", "thumbnail"),
        topk=topk,
    )


async def _search_online_single_query(
    data_id: Any,
    query: str,
    engine: str,
    search_api_key: str,
    topk: int,
) -> List[Dict[str, str]]:
    
    search_results = None
    if engine == "serper":
        search_results = await get_serper_image_response(query) 
        cached_items = await parse_serper_image_response(
            data_id=data_id,
            search_results=search_results,
            query=query,
            topk=topk,
        )
    elif engine == "serpapi":
        search_results = await get_serpapi_image_response(query, search_api_key)
        cached_items = await parse_serpapi_image_response(
            data_id=data_id,
            search_results=search_results,
            query=query,
            topk=topk,
        )
    else:
        raise NotImplementedError(f"engine {engine} not implemented")

    await append_jsonl_async(
        LOG_PATH,
        {
            "data_id": _normalize_data_id(data_id),
            "query": query,
            "cached_data": cached_items,
            "search_response": search_results,
        },
    )
    
    return cached_items


async def func_image_search_by_text_query(
    data_id: Any,
    query_list: List[str],
    engine: str = "serper",  
    search_api_key: Optional[str] = None,
    topk: int = 1,
) -> Tuple[List[str], Dict[str, str]]:
    """
    在线文本搜图主入口。即使失败也返回带上下文的总结。
    """
    if not isinstance(query_list, list) or len(query_list) == 0:
        return [], {"result": "[Image Search Summary] No valid search queries were provided. Proceeding with existing context."}

    query_list = [q for q in query_list if isinstance(q, str) and q.strip()]
    queries_str = ", ".join(query_list)

    if not query_list:
        return [], {"result": "[Image Search Summary] The generated queries were empty. Proceeding with existing context."}

    if not search_api_key:
        return [], {"result": f"[Image Search Summary] Wanted to search for '{queries_str}', but API key is missing. Please answer based on text."}

    result_image: List[str] = []
    result_title: List[str] = []

    if engine in {"serpapi", "serper"}:
        query = query_list[0]
        cached_items = await _search_online_single_query(
            data_id=data_id,
            query=query,
            engine=engine,
            search_api_key=search_api_key,
            topk=topk,
        )
        for item in cached_items[:topk]:
            img_path = item.get("cached_images_path")
            title = item.get("cached_title")
            if img_path:
                result_image.append(img_path)
            if title:
                result_title.append(title)
    else:
        return [], {"result": f"[Image Search Summary] Engine '{engine}' not supported. Searched keywords were: '{queries_str}'."}

    dedup_images = list(dict.fromkeys(result_image))[:topk]
    dedup_titles = list(dict.fromkeys(result_title))[:topk]
    titles_str = "; ".join(dedup_titles) if dedup_titles else ""

    if not dedup_images:
        fallback_text = f"[Image Search Summary] Searched for '{queries_str}'."
        if titles_str:
            fallback_text += f" Found related titles [{titles_str}], but failed to download valid images. Please use these titles as textual clues."
        else:
            fallback_text += " No useful visual information or titles were found. Please try a different keyword or rely on your internal knowledge."
        
        return [], {"result": fallback_text}

    return dedup_images, {
        "result": (
            "[Image Search Succeeded] Relevant image(s) have been successfully retrieved. "
            f"The associated title(s are: {titles_str}. "
            "Please look at the attached images for downstream multimodal reasoning."
        )
    }