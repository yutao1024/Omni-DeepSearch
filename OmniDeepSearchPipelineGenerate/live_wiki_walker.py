import aiohttp
import asyncio
import re
import json
import random
import logging
import yaml
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple, Set
import os

logger = logging.getLogger("WikiWalker")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class LiveWikiWalker:
    def __init__(self, llm_client, llm_model: str, config_path="config/prompts.yaml", max_concurrent=5):
        self.llm_client = llm_client
        self.llm_model = llm_model
        config_path = os.path.join(BASE_DIR, config_path)
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.prompts_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            self.prompts_config = {}

        self.headers = {
        'User-Agent': ''
        }
        self.semaphore = asyncio.Semaphore(max_concurrent) 

    async def _safe_get(self, session: aiohttp.ClientSession, url: str, params=None):
        async with self.semaphore:
            for attempt in range(3):
                try:
                    async with session.get(url, params=params, headers=self.headers, timeout=10) as resp:
                        if resp.status == 429:
                            wait_time = 2 ** attempt
                            logger.warning(f"[Rate Limit] 429 for {url}, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        resp.raise_for_status()
                        if "api.php" in url:
                            return await resp.json()
                        return await resp.text()
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Request failed after 3 attempts: {url} | {e}")
                    await asyncio.sleep(1)
        return None

    async def _search_wikipedia(self, session: aiohttp.ClientSession, query: str) -> str:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": query,
            "redirects": 1,
            "format": "json",
            "formatversion": 2
        }
        data = await self._safe_get(session, url, params)
        if data and "query" in data:
            pages = data["query"].get("pages", [])
            if pages and not pages[0].get("missing"):
                return pages[0]["title"]
        
        params_fuzzy = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json"
        }
        data_fuzzy = await self._safe_get(session, url, params_fuzzy)
        if data_fuzzy and "query" in data_fuzzy:
            search_results = data_fuzzy["query"].get("search", [])
            if search_results:
                return search_results[0]["title"]
                
        return query

    async def _fetch_live_wiki_page(self, session: aiohttp.ClientSession, title: str) -> Tuple[str, List[Dict]]:
        """异步抓取并解析维基百科页面内容和锚点"""
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        html = await self._safe_get(session, url)
        if not html:
            return "", []

        try:
            soup = BeautifulSoup(html, 'html.parser')
            content_div = soup.find(id="mw-content-text")
            if not content_div:
                return "", []

            parser_output = content_div.find(class_="mw-parser-output")
            if not parser_output:
                return "", []

            for unwanted in parser_output.find_all(['div', 'table'], class_=['toc', 'reflist', 'navbox', 'infobox', 'metadata']):
                unwanted.decompose()

            text_snippets = []
            anchors = []
            char_count = 0

            for element in parser_output.find_all(['p', 'li']):
                if len(element.get_text(strip=True)) < 5:
                    continue
                if element.find_parent(id=re.compile(r'cite_note|reflist|references')):
                    continue
                for sup in element.find_all('sup', class_='reference'):
                    sup.decompose()

                paragraph_text = element.get_text(strip=True)
                text_snippets.append(paragraph_text)

                for a in element.find_all('a', href=True):
                    href = a['href']
                    if href.startswith('/wiki/') and ':' not in href: 
                        target_title = self._normalize_title(href.replace('/wiki/', ''))
                        anchor_text = a.get_text(strip=True)
                        anchors.append({"href": target_title, "text": anchor_text})

                char_count += len(paragraph_text)
                if char_count > 2000:
                    break 

            return "\n\n".join(text_snippets), anchors
        except Exception as e:
            logger.error(f"[Parse Error] {title}: {e}")
            return "", []

    async def query_llm(self, sys_prompt: str, user_prompt: str) -> str:
        """异步调用多模态或文本 LLM"""
        if not self.llm_client:
            return '0'
        
        for attempt in range(4):
            try:
                response_text = await self.llm_client.agenerate(system_prompt=sys_prompt, user_prompt=user_prompt)
                if not response_text or str(response_text).startswith("Error:"): 
                    raise Exception(f"Invalid LLM Response: {response_text}")
                return response_text
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(f"[LLM Async Retry] Attempt {attempt+1} failed: {e}. Waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        return '0'

    def _normalize_title(self, s: str) -> str:
        return (s or "").replace("%20", " ").replace("_", " ").replace("%23", "#").strip()

    def _filter_anchors(self, anchors: List[Dict]) -> List[Dict]:
        keep_anchors = []
        broad_stopwords = {
            "united states", "world war ii", "english", "french", "music", 
            "human", "animal", "earth", "water", "science", "art", "history", 
            "country", "city", "state", "film", "book", "united kingdom", "youtube", "google"
        }
        for a in anchors:
            title = a["href"]
            title_lower = title.lower()
            if "(disambiguation)" in title_lower or title_lower.startswith("list of "):
                continue
            if title_lower in broad_stopwords:
                continue
            if title.islower() and len(title.split()) <= 2:
                continue
            keep_anchors.append(a)
        return keep_anchors

    async def _llm_choose_next_topic(self, current_title: str, current_text: str, anchors: List[Dict], visited: Set[str]) -> List[Dict]:
        if not self.llm_client or not anchors:
            return [random.choice(anchors)] if anchors else []
            
        valid_anchors = [a for a in anchors if a["href"].lower() not in visited]
        if not valid_anchors:
            return []
            
        random.shuffle(valid_anchors)
        candidates = valid_anchors[:12]

        candidates_text = "".join([f"[{i}] Link Text: '{a['text']}' -> Target: '{a['href']}'\n" for i, a in enumerate(candidates)])
        
        prompt_cfg = self.prompts_config.get("text_walk", {})
        sys_prompt = prompt_cfg.get("system", "You are an expert finding obscure facts.")
        user_prompt_tpl = prompt_cfg.get("user", "Context:\n{context}\nCandidates:\n{candidates}\nChoose by index.")
        
        context_snippet = current_text[:1000] + "..." if len(current_text) > 1000 else current_text
        user_prompt = user_prompt_tpl.format(title=current_title, context=context_snippet, candidates=candidates_text)

        raw_response = await self.query_llm(sys_prompt, user_prompt)
        
        match = re.search(r'<choice>\s*([\d\s,]+)\s*</choice>', raw_response, re.IGNORECASE)
        if match:
            try:
                indices = [int(i.strip()) for i in match.group(1).split(',')]
                selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
                if selected: return selected
            except: pass

        nums = re.findall(r'\b[0-9]+\b', raw_response)
        if nums:
            try:
                indices = [int(n) for n in nums]
                selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
                if selected: return selected
            except: pass
            
        return [random.choice(candidates)]

    async def _llm_choose_next_topic_for_image(self, current_title, current_text, anchors, visited):
        prompt_cfg = self.prompts_config.get("image_walk", {})
        sys_prompt = prompt_cfg.get("system", "You are a Visual Scavenger Hunter")
        user_prompt_tpl = prompt_cfg.get("user", "[CURRENT TOPIC]: {title}\n...")
        
        valid_anchors = [a for a in anchors if a["href"].lower() not in visited]
        random.shuffle(valid_anchors)
        candidates = valid_anchors[:12]
        
        candidates_text = "".join([f"[{i}] Link Text: '{a['text']}' -> Target: '{a['href']}'\n" for i, a in enumerate(candidates)])
        user_prompt = user_prompt_tpl.format(title=current_title, context=current_text[:1000], candidates=candidates_text)

        raw_response = await self.query_llm(sys_prompt, user_prompt)
        match = re.search(r'<choice>\s*([\d\s,]+)\s*</choice>', raw_response, re.IGNORECASE)
        if match:
            try:
                indices = [int(i.strip()) for i in match.group(1).split(',')]
                return [candidates[i] for i in indices if 0 <= i < len(candidates)]
            except: pass
        return [random.choice(candidates)]


    async def walk_for_image(self, start_title: str, steps: int = 8) -> Optional[Dict]:
        return await self._internal_walk(start_title, steps, use_image_logic=True)
    async def walk(self, start_title: str, steps: int = 5) -> Optional[Dict]:

        return await self._internal_walk(start_title, steps, use_image_logic=False)

    async def _internal_walk(self, start_title: str, steps: int, use_image_logic: bool) -> Optional[Dict]:
        async with aiohttp.ClientSession() as session:
            actual_start_title = await self._search_wikipedia(session, start_title)
            if not actual_start_title: return None
            current_title = self._normalize_title(actual_start_title)
            path, visited = [], {current_title.lower()}

            while len(path) < steps:
                raw_text, raw_anchors = await self._fetch_live_wiki_page(session, current_title)
                if not raw_text: break
                
                current_node = {"title": current_title, "text": raw_text}
                path.append(current_node)

                if use_image_logic and len(path) >= 3:
                    candidate_images = await self._fetch_wiki_images(session, current_title, max_candidates=3)
                    if candidate_images:
                        is_passed, chosen_img = await self._is_gold_mine_image(
                            current_title, candidate_images, current_node["text"]
                        )
                        if is_passed:
                            path[-1]["chosen_image"] = chosen_img 
                            return {"start_entity": start_title, "path": path}

                if len(path) == steps: break

                clean_anchors = self._filter_anchors(raw_anchors)
                if use_image_logic:
                    candidate_anchors = await self._llm_choose_next_topic_for_image(current_title, raw_text, clean_anchors, visited)
                else:
                    candidate_anchors = await self._llm_choose_next_topic(current_title, raw_text, clean_anchors, visited)
                
                if not candidate_anchors:
                    break
                    
                found_next_valid_node = False
                for chosen_anchor in candidate_anchors:
                    next_title = chosen_anchor["href"]
                    test_text, test_anchors = await self._fetch_live_wiki_page(session, next_title)
                    
                    if test_text and len(test_anchors) > 2: 
                        current_title = next_title
                        visited.add(current_title.lower())
                        found_next_valid_node = True
                        break
                    else:
                        visited.add(next_title.lower()) 

                if not found_next_valid_node:
                    break
                
                await asyncio.sleep(0.1)

            min_steps_required = 1
            if len(path) >= min_steps_required:
                return {"start_entity": start_title, "path": path}
            return None