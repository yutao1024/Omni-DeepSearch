import json
import os
import logging
import asyncio
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("KeyManager")

class MultiKeyManager:
    def __init__(self, db_path="data_workspace/api_keys_status.json"):
        self.db_path = db_path
        self.registry = {
            "serper": [],
            "jina": []
        }
        self.load_status()
        self._lock = asyncio.Lock()
        self._search_cache = {}
        self.search_semaphore = asyncio.Semaphore(8)

    def load_status(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    saved_data = json.load(f)
                    for svc, keys in saved_data.items():
                        if svc in self.registry:
                            status_map = {item['key']: item['status'] for item in keys}
                            for item in self.registry[svc]:
                                if item['key'] in status_map:
                                    item['status'] = status_map[item['key']]
            except Exception as e:
                logger.error(f"Error loading JSON status: {e}")

    def save_status(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.registry, f, indent=4)

    async def get_active_key(self, service: str):
        async with self._lock:
            active_keys = [i["key"] for i in self.registry.get(service, []) if i["status"] == "active"]
            if not active_keys:
                logger.critical(f"❌ FATAL: SERVICE [{service}] KEY POOL EMPTY!")
                os._exit(1)
            return active_keys[0]

    async def mark_exhausted(self, service: str, key: str):
        async with self._lock:
            for item in self.registry.get(service, []):
                if item["key"] == key:
                    item["status"] = "exhausted"
                    logger.critical(f"💀 KEY EXHAUSTED: {service} -> {key[:8]}")
                    self.save_status()
                    break


    async def report_api_error(self, service: str, key: str, status: int, body: dict, query: str):

        async with self._lock:
            logger.error(f"""
---------- [API ERROR AUDIT] ----------
Service: {service}
Key: {key[:8]}...
Status Code: {status}
Response Body: {json.dumps(body)}
Trigger Query: {query}
---------------------------------------
""")
            
            if status == 429:
                logger.warning(f"⚠️ Key {key[:6]} is RATE LIMITED. Do NOT rotate. Waiting...")
                return "RETRY_WAIT"

            is_exhausted = False
            if status in [402, 403]:
                is_exhausted = True
            elif "credit" in str(body).lower() or "balance" in str(body).lower():
                is_exhausted = True

            if is_exhausted:
                active_keys = [i for i in self.registry.get(service, []) if i["status"] == "active"]
                if active_keys and active_keys[0]["key"] == key:
                    active_keys[0]["status"] = "exhausted"
                    logger.critical(f"💀 KEY EXHAUSTED CONFIRMED. Status: {status}. Rotating to next...")
                    self.save_status()
                return "ROTATE"
            
            return "UNKNOWN_ERROR"

    def check_cache(self, service, query):
        return self._search_cache.get(f"{service}:{query}")

    def update_cache(self, service, query, result):
        self._search_cache[f"{service}:{query}"] = result

key_manager = MultiKeyManager()