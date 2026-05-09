import json
import os
import re
import asyncio
import aiohttp
from typing import Any, Dict, List, Optional, Tuple

from serpapi import GoogleSearch
from llm_provider import get_llm_provider
from async_utils import async_retry, run_in_thread
from key_manager import key_manager
import logging
logger = logging.getLogger(__name__)

LOCAL_TIMEOUT = 60
WEB_TIMEOUT = 60
MAX_PAGE_CHARS = 3000   

def _normalize_query(query: str) -> str:
    return query.strip()

def _cache_key(engine: str, query: str) -> str:
    return f"{engine}::{_normalize_query(query)}"

def clean_content(text: str) -> str:
    if not text: return ""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[|#*=-]{3,}', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def _safe_yes_no(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if re.fullmatch(r"\s*yes\s*[\.\!]*\s*", text, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\s*no\s*[\.\!]*\s*", text, flags=re.IGNORECASE):
        return False

    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return bool(m and m.group(1).lower() == "yes")

async def _call_judge_model(system_prompt, user_prompt, config):
    try:
        provider = config['llm'].get('filter_provider', config['llm'].get('filter_provider'))
        llm = get_llm_provider(config['llm'], provider)
        res = await llm.agenerate(system_prompt=system_prompt, user_prompt=user_prompt)
        return res if res else ""
    except Exception as e:
        print(f"[WARN] Failed to call judge model: {e}")
        return ""

async def summarize_webpage(original_question: str, current_thought: str, query: str, webpage_content: str, config: dict) -> str:
    cleaned_text = clean_content(webpage_content)[:MAX_PAGE_CHARS]

    system_prompt = """You are an expert information summarizer. 
Your task is to extract information from a webpage to help an agent solve a complex multi-hop challenge. Even if the webpage lacks the direct answer, you must provide a general summary of its content."""
    
    user_prompt = f"""
### GLOBAL CONTEXT
[Original Question]: {original_question}
[Agent's Current Reasoning]: {current_thought}
[Current Search Step]: {query}

### WEBPAGE CONTENT
{cleaned_text}

### TASK
Based on the Global Context, extract and summarize the specific facts from the webpage that help answer the original question or support the current reasoning step.
- Be extremely precise with names, dates, and technical details.
- If the page contains the final answer or a crucial "bridge" entity, highlight it.
- IMPORTANT: If the page does not contain useful information, DO NOT return an empty string. Instead, summarize what the page is actually about and provide any vaguely related context.
- Return the relevant info as concise plain text.
"""
    return await _call_judge_model(system_prompt, user_prompt, config)


async def call_serpapi(query: str, topk: int, search_api_key: str) -> List[Dict[str, str]]:
    def _sync_search():
        params = {
            "q": query,
            "google_domain": "google.com",
            "api_key": search_api_key,
        }
        try:
            search_obj = GoogleSearch(params)
            return search_obj.get_dict()
        except Exception as e:
            print(f"[WARN] call_serpapi failed: {e}")
            return {}

    raw_results = await run_in_thread(_sync_search)
    organic_results = raw_results.get("organic_results", [])
    if not organic_results:
        return []
    return [
        {
            "link": item.get("link"),
            "title": item.get("title"),
            "snippet": item.get("snippet", "")
        } 
        for item in organic_results[:topk] if item.get("link")
    ]


@async_retry(max_retries=3)
async def call_serper(query: str, topk: int = 3) -> List[Dict[str, str]]:
    cached = key_manager.check_cache("serper_text", query)
    if cached: return cached

    async with key_manager.search_semaphore:
        while True:
            current_key = await key_manager.get_active_key("serper")
            headers = {"X-API-KEY": current_key, "Content-Type": "application/json"}
            url = "https://google.serper.dev/search"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={"q": query}, headers=headers, timeout=15) as resp:
                        status_code = resp.status
                        try:
                            data = await resp.json()
                        except:
                            data = {"raw_text": await resp.text()}

                        if status_code == 200:
                            organic = data.get("organic", [])
                            res = [{"link": i.get("link"), "title": i.get("title"), "snippet": i.get("snippet")} for i in organic[:topk]]
                            key_manager.update_cache("serper_text", query, res)
                            return res

                        action = await key_manager.report_api_error("serper", current_key, status_code, data, query)
                        if action == "RETRY_WAIT":
                            await asyncio.sleep(5)
                            continue
                        elif action == "ROTATE":
                            continue
                        else:
                            break
            except Exception as e:
                logger.error(f"Text Search Network Error: {e}")
                await asyncio.sleep(2)
                break
    return []

async def is_snippet_useful(original_question: str, current_thought: str, query: str, title: str, snippet: str, config: dict) -> bool:
    if not title and not snippet:
        return False
        
    system_prompt = "You are a search result auditor. Decide if this link should be filtered out. Only filter out obvious ads or complete garbage."

    user_prompt = f"""
### GLOBAL CONTEXT
[Original Question]: {original_question}
[Agent's Reasoning]: {current_thought}
[Current Query]: {query}

### SEARCH RESULT
[Title]: {title}
[Snippet]: {snippet}

### TASK
Evaluate the search result. 
Output "No" ONLY if the result is clearly an advertisement, spam, or entirely irrelevant/useless to the context. 
Otherwise, output "Yes" to keep it.
Output ONLY "Yes" or "No"."""
    
    response = await _call_judge_model(system_prompt, user_prompt, config)
    return _safe_yes_no(response)

async def _fetch_jina_pages(links: List[str], titles: List[str], **kwargs) -> List[str]:
    results = []
    
    async def _fetch_one(link, title):
        while True:
            current_key = await key_manager.get_active_key("jina")
            if not current_key: return None
            
            headers = {"Authorization": f"Bearer {current_key}"}
            url = f"https://r.jina.ai/{link}"
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url, timeout=30) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            return f"{title}\n{text}"
                        elif resp.status == 402: 
                            await key_manager.mark_exhausted("jina", current_key)
                            continue 
                        else:
                            return None
            except: return None

    for link, title in zip(links, titles):
        res = await _fetch_one(link, title)
        if res: results.append(res)
    return results

def _format_web_doc(doc_item: str) -> str:
    doc_item = doc_item[:MAX_PAGE_CHARS]
    lines = doc_item.split("\n")
    title = lines[0] if lines else "No Title"
    text = "\n".join(lines[1:]) if len(lines) > 1 else ""
    return f"(Title: {title}) {text}"

async def fetch_webpage(query: str, config: dict, topk: int = 3, engine: str = "serper") -> List[Any]:
    query = _normalize_query(query)
    search_api_key = config.get('serper_api_key') if engine == "serper" else config.get('serpapi_api_key')
    jina_api_key = config.get('jina_api_key')

    if not search_api_key: return []

    if engine == "serpapi":
        search_results = await call_serpapi(query, topk, search_api_key)
    else:
        search_results = await call_serper(query, topk) 

    if not search_results: return []

    links = [r['link'] for r in search_results]
    titles = [r['title'] for r in search_results]
    return await _fetch_jina_pages(links, titles, jina_api_key)

async def search(query_list: List[str], config: dict, original_question: str, current_thought: str, engine: str = "serper", topk: int = 3) -> Dict[str, str]:
    processed_links = set()
    pretty_results: List[str] = []
    fallback_snippets: List[str] = []  
    
    search_api_key = config.get('serper_api_key') if engine == "serper" else config.get('serpapi_api_key')
    jina_api_key = config.get('jina_api_key')

    if not search_api_key:
        return {"result": "Missing search API key."}

    if not query_list:
        return {"result": "Empty query list, no search performed."}
    query = _normalize_query(query_list[0])

    if engine == "serpapi":
        search_results = await call_serpapi(query, topk, search_api_key)
    else:
        search_results = await call_serper(query, topk) 

    format_reference = ""
    idx = 1
    if search_results:
        for item in search_results:
            link = item.get('link')
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            
            if title or snippet:
                fallback_snippets.append(f"Title: {title}\nSnippet: {snippet}")

            if not link or link in processed_links:
                continue
            
            if not await is_snippet_useful(original_question, current_thought, query, title, snippet, config):
                continue

            processed_links.add(link)
            full_content_list = await _fetch_jina_pages([link], [title], jina_api_key)
            if not full_content_list:
                continue
            
            full_text = full_content_list[0]
            summarized_doc = await summarize_webpage(original_question, current_thought, query, full_text, config)
            
            if summarized_doc.strip():
                format_reference += f"Doc {idx}: {summarized_doc}\n"
            else:
                format_reference += f"Doc {idx}: (No detailed content extracted) Title: {title}\nSnippet: {snippet}\n"
            idx += 1

        if format_reference:
            pretty_results.append(format_reference)

    if pretty_results:
        return {"result": "\n---\n".join(pretty_results)}
        
    elif fallback_snippets:
        return {"result": "Detailed webpage extraction yielded no results. Here are the raw search snippets instead:\n" + "\n---\n".join(fallback_snippets)}
    
    return {"result": "Search engine returned no results at all. Please try different keywords."}