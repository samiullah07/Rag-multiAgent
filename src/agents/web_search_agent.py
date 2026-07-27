"""Web search agent using Tavily API.

Provides a single function `web_search(query: str, max_results: int = 5) -> list[dict]`
that returns a list of result dictionaries with keys `title`, `url`, `content`.
"""

import os
from tavily import TavilyClient
from typing import List, Dict

from src.config.config import settings

def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Perform a web search via Tavily.

    Args:
        query: The search query string.
        max_results: Number of results to return (default 5).
    Returns:
        A list of dictionaries, each containing `title`, `url`, and `content`.
    """
    # Read via the project's Settings object (pydantic-settings loads .env
    # internally and does NOT populate os.environ as a side effect —
    # os.getenv("TAVILY_API_KEY") will return None even with a valid .env
    # file, which is exactly what caused this to fail in the running app
    # while appearing to work in standalone load_dotenv() test scripts).
    # Fall back to os.getenv for anyone running outside this project's
    # config system, but settings.tavily_api_key is the real source of truth.
    api_key = settings.tavily_api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set (checked settings.tavily_api_key and os.environ)")
    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results)
    # response contains a "results" list with dicts; keep only needed fields
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        })
    return results
