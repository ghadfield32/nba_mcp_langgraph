"""Retry utilities for handling rate limits and other temporary failures.

location: app/utils/retries.py
"""

import asyncio
import functools
from typing import (
    Any,
    Callable,
    TypeVar,
)

from app.core.logging import logger

T = TypeVar('T')

def retry_on_ratelimit(max_retries=3, initial_delay=1):
    """
    Decorator to retry functions when rate limited (HTTP 202 response or RateLimitError).
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds (will be doubled for each retry)
        
    Returns:
        A decorator function
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    result = await fn(*args, **kwargs)
                    
                    # Detect a 202 / rate-limit marker in string results
                    if isinstance(result, str) and "202" in result:
                        logger.warning(
                            f"Rate limit detected (202) on attempt {attempt+1}/{max_retries}, "
                            f"retrying in {delay}s"
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                        
                    return result
                    
                except Exception as e:
                    # Check if it's a rate limit error based on exception attributes
                    is_rate_limit = (
                        hasattr(e, "status_code") and e.status_code == 429
                        or "rate limit" in str(e).lower()
                        or "ratelimit" in str(e).lower()
                        or "too many requests" in str(e).lower()
                        or (hasattr(e, "status_code") and e.status_code == 202)
                    )
                    
                    if is_rate_limit and attempt < max_retries - 1:
                        logger.warning(
                            f"Rate limit exception on attempt {attempt+1}/{max_retries}, "
                            f"retrying in {delay}s: {str(e)}"
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                        last_error = e
                        continue
                    else:
                        # Not a rate limit or final attempt
                        raise
            
            # If we've exhausted all retries with rate limits
            if last_error:
                raise last_error
                
            # Return last result even if rate-limited (shouldn't reach here)
            return result
            
        return wrapper
    return decorator 