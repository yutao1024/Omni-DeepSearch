import aiohttp
import asyncio
import re
import logging
import os
import time
import json
import random 
from live_wiki_walker import LiveWikiWalker
from wiki_utils import download_and_resize_image

logger = logging.getLogger("EnhancedWikiWalker")
logger.setLevel(logging.INFO)

class EnhancedWikiWalker(LiveWikiWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used_image_urls = set()

    async def _fetch_wiki_images(self, session: aiohttp.ClientSession, title: str, max_candidates=5):

        valid_images = []
        seen_urls = set()
        logger.info(f">>> [Image Fetch] Starting search for topic: '{title}'")

        if random.random() < 0.6:
            params_main = {
                "action": "query",
                "titles": title,
                "prop": "pageimages",
                "piprop": "original", 
                "format": "json",
                "formatversion": 2,
                "redirects": 1
            }
            
            try:
                data = await self._safe_get(session, "https://en.wikipedia.org/w/api.php", params=params_main)
                if data:
                    page = data.get("query", {}).get("pages", [{}])[0]
                    if "original" in page:
                        img_url = page["original"]["source"]
                        if img_url not in self.used_image_urls:
                            logger.info(f"    [Main Image Found] URL: {img_url}")
                            valid_images.append({
                                "url": img_url,
                                "description": f"Primary high-res image for '{title}'."
                            })
                            seen_urls.add(img_url)
            except Exception as e:
                logger.error(f"    [Main Image Error] {e}")

        params_gallery = {
            "action": "query",
            "titles": title,
            "prop": "images",
            "imlimit": 50, 
            "format": "json",
            "formatversion": 2,
            "redirects": 1
        }
        
        try:
            data_gal = await self._safe_get(session, "https://en.wikipedia.org/w/api.php", params=params_gallery)
            if data_gal:
                raw_images = data_gal.get("query", {}).get("pages", [{}])[0].get("images", [])
                
                random.shuffle(raw_images)

                junk_pattern = re.compile(
                    r'(stub|logo|icon|button|edit|magnify|ambox|portal|star|check|lock|arrow|'
                    r'wikiquote|wikisource|wikinews|wiktionary|speaker|commons-logo)', 
                    re.IGNORECASE
                )

                for img_info in raw_images:
                    if len(valid_images) >= max_candidates:
                        break

                    img_title = img_info["title"]
                    if junk_pattern.search(img_title): continue
                    if not img_title.lower().endswith(('.jpg', '.jpeg', '.png', '.svg')): continue
                    
                    params_info = {
                        "action": "query",
                        "titles": img_title,
                        "prop": "imageinfo",
                        "iiprop": "url|size",
                        "format": "json",
                        "formatversion": 2
                    }
                    data_info = await self._safe_get(session, "https://en.wikipedia.org/w/api.php", params=params_info)
                    if not data_info: continue
                    
                    info_page = data_info.get("query", {}).get("pages", [{}])[0]
                    info = info_page.get("imageinfo", [{}])[0]
                    
                    img_url = info.get("url")
                    if not img_url or img_url in seen_urls or img_url in self.used_image_urls: continue

                    width = info.get("width", 0)
                    height = info.get("height", 0)
                    
                    if not img_title.lower().endswith('.svg'):
                        if width < 500 or height < 500:
                            continue

                    logger.info(f"    [High-Res Added] {img_title} ({width}x{height})")
                    valid_images.append({
                        "url": img_url,
                        "description": f"High-resolution gallery image for '{title}'."
                    })
                    seen_urls.add(img_url)

        except Exception as e:
            logger.warning(f"    [Gallery Error] Failed for {title}: {str(e)}")
        
        return valid_images

    async def _is_gold_mine_image(self, title, image_list, wiki_text, **kwargs):
        storage_dir = kwargs.get("storage_dir") or getattr(self, "audit_storage_dir", "data_workspace/audit_temp")
        os.makedirs(storage_dir, exist_ok=True)

        clean_title = re.sub(r'\W+', '', title)
        timestamp = int(time.time())
        candidate_paths = []

        random.shuffle(image_list)

        for i, img_info in enumerate(image_list[:3]):
            tmp_name = f"audit_{clean_title}_{timestamp}_{i}.jpg"
            target_path = os.path.join(storage_dir, tmp_name)
            saved_path = await download_and_resize_image(img_info['url'], target_path)
            if saved_path:
                candidate_paths.append(saved_path)

        if not candidate_paths:
            return False, None

        auditor_cfg = self.prompts_config.get("image_auditor", {})
        sys_prompt = auditor_cfg.get("system", "")
        user_tpl = auditor_cfg.get("user", "Entity: {title}\nContext: {context}")
        
        enhanced_user_prompt = user_tpl.format(
            title=title, 
            context=wiki_text[:300]
        )

        try:
            logger.info(f"    [Audit LLM] Evaluating {len(candidate_paths)} images for '{title}'...")
            res = await self.llm_client.agenerate(
                system_prompt=sys_prompt,
                user_prompt=enhanced_user_prompt,
                media_files=candidate_paths 
            )
            
            match = re.search(r'\{[\s\S]*\}', res)
            if not match: raise ValueError("Invalid LLM response format")
            decision = json.loads(match.group(0))
            
            status = decision.get("status", "FAIL").upper()

            if status == "PASS":
                best_idx = int(decision.get("best_index", 0))
                best_idx = min(max(0, best_idx), len(candidate_paths)-1)
                
                chosen_url = image_list[best_idx]['url']
                self.used_image_urls.add(chosen_url)

                chosen_path = candidate_paths[best_idx]
                final_name = f"{clean_title}_{timestamp}.jpg"
                final_path = os.path.join(storage_dir, final_name)
                
                await asyncio.to_thread(os.rename, chosen_path, final_path)
                logger.info(f"    [Audit PASS] Kept: {final_path}")

                for i, p in enumerate(candidate_paths):
                    if i != best_idx and os.path.exists(p):
                        await asyncio.to_thread(os.remove, p)
                
                return True, {"url": chosen_url, "local_path": final_path}
            
            else:
                logger.info(f"    [Audit FAIL] Reason: {decision.get('reason', 'N/A')}")
                for p in candidate_paths:
                    if os.path.exists(p): 
                        await asyncio.to_thread(os.remove, p)
                return False, None

        except Exception as e:
            logger.error(f"    [Audit Error] {str(e)}")
            for p in candidate_paths:
                if os.path.exists(p):
                    await asyncio.to_thread(os.remove, p)
            return False, None

    async def walk_with_images(self, start_title, steps=8):
        logger.info(f"========== WALK START: {start_title} ==========")
        
        async with aiohttp.ClientSession() as session:
            path_result = await self.walk_for_image(start_title, steps)
            
            if not path_result:
                return None
                
            final_node = path_result["path"][-1]
            final_node["images"] = await self._fetch_wiki_images(session, final_node["title"], max_candidates=3)
            
            if not final_node["images"]:
                return None
                
            return path_result