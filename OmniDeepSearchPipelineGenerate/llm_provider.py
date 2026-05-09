import os
import json
import base64
import mimetypes
import asyncio
import logging
import copy
import aiohttp
from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError
from async_utils import async_retry, run_in_thread
import time
class LLMEmptyResponseError(Exception):
    pass

logger = logging.getLogger("LLMProvider")

def setup_logging():
    if logger.handlers:
        return
        
    logger.setLevel(logging.DEBUG) 

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(sh)

    fh = logging.FileHandler("llm_provider.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

setup_logging()


def sanitize_log_content(content):
    if isinstance(content, dict):
        return {k: (f"[{k.upper()}_DATA_OMITTED]" if k in ['image_url', 'input_audio', 'image', 'audio', 'data'] 
                else sanitize_log_content(v)) for k, v in content.items()}
    elif isinstance(content, list):
        return [sanitize_log_content(item) for item in content]
    return content

class OpenAICompatibleProvider:
    def __init__(self, llm_config, log_enabled=True, model_override=None, default_model="gpt-4o"):
        self.api_key = llm_config.get('api_key')
        self.base_url = llm_config.get('base_url')
        self.model_name = model_override or llm_config.get('model_name', default_model)
        
        timeout_val = float(llm_config.get('timeout', 150.0))
        self.client = AsyncOpenAI(
            api_key=self.api_key, 
            base_url=self.base_url, 
            timeout=timeout_val
        )
        self.temperature = llm_config.get('temperature', 0.0)
        self.max_tokens = llm_config.get('max_tokens', 8192)
        self.log_enabled = log_enabled

        max_concurrency = llm_config.get('max_concurrency', 10)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _get_base64_encoded_async(self, file_path):
        def _read():
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return await run_in_thread(_read)

    @async_retry(
        max_retries=2, 
        base_delay=5.0, 
        exceptions=(APITimeoutError, APIConnectionError, RateLimitError, LLMEmptyResponseError)
    )
    async def agenerate(self, system_prompt: str = None, user_prompt: str = None, media_files: list = None, messages: list = None) -> str:
        async with self.semaphore:
            if messages:
                final_messages = messages
            else:
                final_messages = []
                if system_prompt:
                    final_messages.append({"role": "system", "content": system_prompt})
                
                user_content = []
                if media_files:
                    for path in media_files:
                        if not path or not os.path.exists(path): continue
                        ext = path.lower()
                        b64_data = await self._get_base64_encoded_async(path)
                        
                        if ext.endswith((".wav", ".mp3", ".ogg")):
                            user_content.append({"type": "input_audio", "input_audio": {"data": b64_data, "format": "wav"}})
                        elif ext.endswith((".jpg", ".jpeg", ".png")):
                            mime_type, _ = mimetypes.guess_type(path)
                            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type or 'image/jpeg'};base64,{b64_data}"}})
                
                if user_prompt:
                    user_content.append({"type": "text", "text": user_prompt})
                
                if user_content:
                    final_messages.append({"role": "user", "content": user_content})

            if self.log_enabled:
                logger.info(f"--- [{self.__class__.__name__} Request] ---\n{json.dumps(sanitize_log_content(final_messages), indent=2, ensure_ascii=False)}")

            extra_body = {}
            if "gemini" in self.model_name.lower():
                extra_body = {
                    "safety_settings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
                    ]
                }

            response = await self.client.chat.completions.create(
                model=self.model_name, 
                messages=final_messages,
                max_tokens=self.max_tokens, 
                temperature=self.temperature,
                extra_body=extra_body
            )
            full_response_dict = response.model_dump() 
            
            choice = response.choices[0]
            res_text = choice.message.content
            finish_reason = choice.finish_reason
            tool_calls = getattr(choice.message, 'tool_calls', None)
            
            if self.log_enabled:
                logger.info(f"--- [{self.__class__.__name__} Response] ---\n{res_text}\n")
            return res_text

    def generate(self, *args, **kwargs):
        return asyncio.run(self.agenerate(*args, **kwargs))


class GeminiDubrifyProvider(OpenAICompatibleProvider):
    def __init__(self, llm_config, log_enabled=True, model_override=None):
        super().__init__(llm_config, log_enabled, model_override, default_model='gemini-2.0-flash-thinking-exp')

class GPTDubrifyProvider(OpenAICompatibleProvider):
    def __init__(self, llm_config, log_enabled=True, model_override=None):
        super().__init__(llm_config, log_enabled, model_override, default_model='gpt-4o')



class QwenOmniProvider(OpenAICompatibleProvider):
    def __init__(self, llm_config, log_enabled=True, model_override=None):
        super().__init__(llm_config, log_enabled, model_override, default_model='qwen3.5-omni-plus')
        self.max_concurrency = llm_config.get('max_concurrency', 1) 
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    @async_retry(
        max_retries=10,       
        base_delay=15.0,      
        exceptions=(RateLimitError, APIConnectionError, APITimeoutError, Exception)
    )
    async def agenerate(self, system_prompt: str = None, user_prompt: str = None, media_files: list = None, messages: list = None) -> str:
        async with self.semaphore:
            final_messages = copy.deepcopy(messages) if messages else []
            
            if not messages:
                if system_prompt:
                    final_messages.append({"role": "system", "content": system_prompt})
                
                user_content = []
                if media_files:
                    for path in media_files:
                        if not path or not os.path.exists(path): continue
                        ext = path.lower()
                        b64_data = await self._get_base64_encoded_async(path)
                        b64_data = b64_data.replace('\n', '').replace('\r', '') 
                        
                        if ext.endswith((".wav", ".mp3", ".ogg")):
                            user_content.append({
                                "type": "input_audio", 
                                "input_audio": {
                                    "data": b64_data, 

                                    "format": "mp3" if ext.endswith(".mp3") else "wav"
                                }
                            })
                        elif ext.endswith((".jpg", ".jpeg", ".png")):
                            mime_type, _ = mimetypes.guess_type(path)
                            user_content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:{mime_type or 'image/jpeg'};base64,{b64_data}"}
                            })
                
                if user_prompt:
                    user_content.append({"type": "text", "text": user_prompt})
                if user_content:
                    final_messages.append({"role": "user", "content": user_content})

            for msg in final_messages:
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "input_audio":
                            audio_info = item.get("input_audio", {})
                            raw_data = audio_info.get("data", "")
                            
                            pure_b64 = raw_data.split(",")[-1] if "," in raw_data else raw_data
                            pure_b64 = pure_b64.replace('\n', '').replace('\r', '')
                            
                            audio_info["data"] = f"data:;base64,{pure_b64}"
                        
                        elif item.get("type") == "image_url":
                            img_info = item.get("image_url", {})
                            url_str = img_info.get("url", "")
                            if url_str and not url_str.startswith("data:") and not url_str.startswith("http"):
                                img_info["url"] = f"data:image/jpeg;base64,{url_str}"
                            img_info["url"] = img_info["url"].replace('\n', '').replace('\r', '')

            full_text = ""
            try:
                if self.log_enabled:
                    logger.info(f"--- [Qwen Request] Base64 formatted to Ali-spec (data:;base64, for audio) ---")

                response_stream = await self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=final_messages,
                    max_tokens=self.max_tokens, 
                    temperature=self.temperature,
                    stream=True,  
                    modalities=["text"],
                    stream_options={"include_usage": True}
                )
                
                async for chunk in response_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            full_text += delta.content
                
                if not full_text:
                    logger.warning("Qwen returned an empty response, possibly blocked by safety filters.")

                return full_text

            except RateLimitError as e:
                logger.warning(f"[Qwen RateLimit] 429 Detected. Backing off... Msg: {e.message}")
                raise e 
            except Exception as e:
                err_msg = str(e).lower()
                if any(k in err_msg for k in ["429", "quota", "limit", "throttling", "too many"]):
                    logger.warning(f" [Qwen Throttled] Waiting for quota reset... Error: {e}")
                raise e
            
class XiaomiMiMoProvider(OpenAICompatibleProvider):
    def __init__(self, llm_config, log_enabled=True, model_override=None):
        super().__init__(llm_config, log_enabled, model_override, default_model='xiaomi/mimo-v2.5')

def get_llm_provider(llm_config_root, llm_name):
    mapping = {
        'gpt-5.4': 'gpt',
        'gemini-3-pro': 'gemini',
        'claude-sonnet-4-6': 'claude',
        'mimo-v2.5': 'mimo',    
        'xiaomi/mimo-v2.5': 'mimo' 
    }
    
    provider_key = mapping.get(llm_name)
    
    if not provider_key:
        if 'mimo' in llm_name.lower(): provider_key = 'mimo'
        elif 'gpt' in llm_name: provider_key = 'gpt'
        elif 'gemini' in llm_name: provider_key = 'gemini'
        elif 'qwen' in llm_name: provider_key = 'qwen'
        elif 'claude' in llm_name: provider_key = 'claude'
        else: raise ValueError(f"Unsupported provider: {llm_name}")

    config = llm_config_root.get(provider_key)
    if not config:
        raise ValueError(f"Config for provider key '{provider_key}' not found in configuration.")

    model_override = llm_name if llm_name != provider_key else None

    if provider_key == 'mimo':
        return XiaomiMiMoProvider(config, model_override=model_override)
    elif provider_key == 'qwen':
        return QwenOmniProvider(config, model_override=model_override)
    elif provider_key == 'gemini':
        return GeminiDubrifyProvider(config, model_override=model_override)
    elif provider_key == 'gpt':
        return GPTDubrifyProvider(config, model_override=model_override)
    else:
        return OpenAICompatibleProvider(config, model_override=model_override)