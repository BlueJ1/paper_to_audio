"""Selected-provider construction without importing optional providers eagerly."""
import os

DEFAULT_LLM_MODEL = "gemma-3-27b-it"
DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b"


def build_llm(model=None, provider="google"):
    if provider not in {"google", "cerebras"}:
        raise ValueError("Unknown LLM provider")
    key_name = "GOOGLE_API_KEY" if provider == "google" else "CEREBRAS_API_KEY"
    key = os.getenv(key_name, "").strip()
    if not key:
        raise RuntimeError(f"Missing {key_name} in environment.")
    if provider == "cerebras":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or DEFAULT_CEREBRAS_MODEL,
                          base_url="https://api.cerebras.ai/v1", api_key=key, temperature=0.1, timeout=120, max_retries=1)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model or DEFAULT_LLM_MODEL,
                                  google_api_key=key, vertexai=False, temperature=0.1, timeout=120, max_retries=1)


def wrap_langchain_llm(chat_model):
    def call(prompt):
        response = chat_model.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        raise ValueError("Provider returned no text content")
    return call
