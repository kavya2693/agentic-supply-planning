"""Optional language model for the planner briefing. Uses OpenRouter if a key is set,
otherwise a local Ollama server, otherwise none (the workflow still runs end to end)."""

import os
import urllib.request


def get_llm():
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr
    except ImportError:
        return None
    if os.environ.get("OPENROUTER_API_KEY"):
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        )
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "phi4"),
            temperature=0,
            base_url="http://localhost:11434/v1",
            api_key=SecretStr("ollama"),
        )
    except OSError:
        return None
