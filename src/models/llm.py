import groq
from langchain_groq import ChatGroq
import re
import time
from src.config.config import settings
def get_chat_llm():
    """
    Returns a LangChain ChatModel instance configured to use Groq.
    Includes a retry wrapper for Groq rate limit errors that parses the
    suggested wait time from error messages and retries with that delay.
    This provides resilience against token quota limits for evaluation runs.
    """
    # Base LLM from LangChain
    base_llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
    )

    # Wrapper class that handles Groq rate limit errors with precise retry timing
    class LLMWithRetry:
        def __init__(self, llm):
            self._llm = llm

        def invoke(self, prompt, **kwargs):
            max_attempts = 2
            for attempt in range(max_attempts + 1):
                try:
                    return self._llm.invoke(prompt, **kwargs)
                except groq.RateLimitError as exc:
                    if attempt == max_attempts:
                        raise
                    msg = str(exc)
                    m = re.search(r"try again in (?:(\d+)m)?(\d+\.?\d*)s", msg)
                    if m:
                        minutes = float(m.group(1)) if m.group(1) else 0.0
                        seconds = float(m.group(2))
                        wait = minutes * 60 + seconds
                    else:
                        wait = 15.0
                    time.sleep(wait + 1)

        def __getattr__(self, name):
            return getattr(self._llm, name)

    return LLMWithRetry(base_llm)