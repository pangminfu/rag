"""LLM and embedding provider abstraction.

Reads provider + model from env vars and returns a LangChain chat or embedding
model. Add a new provider by extending the dispatch in chat() / embeddings().

Required env vars:
    LLM_PROVIDER         ollama | openai | gemini
    LLM_MODEL            provider-specific model name
    EMBEDDING_PROVIDER   ollama | openai | gemini
    EMBEDDING_MODEL      provider-specific model name

Provider credentials (only the ones you actually use):
    OLLAMA_HOST          base URL, e.g. http://192.168.1.5:11434
    OPENAI_API_KEY       OpenAI API key
    GOOGLE_API_KEY       Google AI Studio key (for Gemini)
"""

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def _env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


class ModelProvider:
    def __init__(self):
        self.llm_provider = _env("LLM_PROVIDER").lower()
        self.llm_model = _env("LLM_MODEL")
        self.embedding_provider = _env("EMBEDDING_PROVIDER").lower()
        self.embedding_model = _env("EMBEDDING_MODEL")

    def chat(self) -> BaseChatModel:
        provider, model = self.llm_provider, self.llm_model

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, base_url=_env("OLLAMA_HOST"))

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            _env("OPENAI_API_KEY")
            return ChatOpenAI(model=model)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            _env("GOOGLE_API_KEY")
            return ChatGoogleGenerativeAI(model=model)

        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. Expected ollama | openai | gemini."
        )

    def embeddings(self) -> Embeddings:
        provider, model = self.embedding_provider, self.embedding_model

        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=model, base_url=_env("OLLAMA_HOST"))

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            _env("OPENAI_API_KEY")
            return OpenAIEmbeddings(model=model)

        if provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _env("GOOGLE_API_KEY")
            return GoogleGenerativeAIEmbeddings(model=model)

        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: {provider!r}. Expected ollama | openai | gemini."
        )
