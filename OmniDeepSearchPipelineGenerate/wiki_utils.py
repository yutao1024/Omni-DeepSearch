import aiohttp
import asyncio
from io import BytesIO
from PIL import Image
import logging
import os

try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False

logger = logging.getLogger("wiki_utils")

def _sync_process_image(file_content, image_url, save_path, max_image_pixels):
    """保留原有的 PIL 处理逻辑，但在线程中运行"""
    if image_url.lower().endswith('.svg'):
        if not HAS_CAIROSVG: 
            return None
        file_content = cairosvg.svg2png(bytestring=file_content)

    image = Image.open(BytesIO(file_content))
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGBA")
        new_img = Image.new("RGB", image.size, (255, 255, 255))
        new_img.paste(image, mask=image.split()[3]) 
        image = new_img
    elif image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    cur_pixels = width * height
    if cur_pixels > max_image_pixels:
        scale = (max_image_pixels / cur_pixels) ** 0.5
        image = image.resize((int(width * scale), int(height * scale)), 
                             resample=getattr(Image, 'Resampling', Image).LANCZOS)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path, format="JPEG", quality=85)
    return save_path

async def download_and_resize_image(image_url: str, save_path: str, max_image_pixels: int = 1024 * 1024) -> str:
    max_retries = 3
    base_delay = 2
    headers = {}

    async with aiohttp.ClientSession(headers=headers) as session:
        for attempt in range(max_retries):
            try:
                async with session.get(image_url, timeout=15) as response:
                    if response.status == 429:
                        wait_time = base_delay * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    file_content = await response.read()
                    
                    return await asyncio.to_thread(_sync_process_image, file_content, image_url, save_path, max_image_pixels)

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed {image_url}: {e}")
                await asyncio.sleep(base_delay)
    return None