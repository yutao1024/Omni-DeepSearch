import asyncio
import logging
import random
from functools import wraps

logger = logging.getLogger("AsyncUtils")

def async_retry(max_retries=3, base_delay=1.0, max_delay=15.0, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"[Retry Failed] {func.__name__} failed after {max_retries} attempts. Error: {e}")
                        raise e
                    
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(f"[Retry {attempt+1}/{max_retries}] {func.__name__} error: {e}. Waiting {delay:.2f}s...")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

async def run_in_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)