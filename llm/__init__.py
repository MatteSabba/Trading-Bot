"""Package per l'integrazione con server LLM locali (Ollama / vLLM) e fallback euristico."""

from llm.local_client import LocalLLMClient

__all__ = ["LocalLLMClient"]