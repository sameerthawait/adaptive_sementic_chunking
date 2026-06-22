"""Embedding module for Adaptive Semantic Chunking.

Wraps Ollama's /api/embed endpoint as a LangChain-compatible Embeddings class.
"""

import logging
from typing import Any
import httpx
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class OllamaEmbedder(Embeddings):
    """Custom LangChain Embeddings class wrapping local Ollama's embedding API."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initializes the embedder.

        Args:
            model: The name of the local Ollama embedding model.
            ollama_url: The base URL of the Ollama server.
            http_client: Optional async client for dependency injection.
        """
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self._external_client = http_client

    def _get_client(self) -> httpx.AsyncClient:
        """Returns the async client."""
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=30.0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronously embeds a list of documents.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vector lists.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.aembed_documents(texts))
            else:
                return loop.run_until_complete(self.aembed_documents(texts))
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.aembed_documents(texts))
            finally:
                new_loop.close()

    def embed_query(self, text: str) -> list[float]:
        """Synchronously embeds a single query.

        Args:
            text: Query string.

        Returns:
            The query embedding vector.
        """
        res = self.embed_documents([text])
        return res[0] if res else []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embeds a list of documents.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vector lists.
        """
        if not texts:
            return []

        url = f"{self.ollama_url}/api/embed"
        # If the input list is very large, chunk it to avoid exceeding Ollama payload limits
        chunk_size = 32
        embeddings = []
        client = self._get_client()

        for i in range(0, len(texts), chunk_size):
            batch = texts[i:i + chunk_size]
            payload = {
                "model": self.model,
                "input": batch,
            }
            try:
                if self._external_client is not None:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                else:
                    async with client as active_client:
                        response = await active_client.post(url, json=payload)
                        response.raise_for_status()
                        data = response.json()

                batch_embeddings = data.get("embeddings")
                if not batch_embeddings:
                    raise ValueError(f"Ollama API did not return embeddings: {data}")
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings via Ollama: {e}")
                # Return zero vector fallback to prevent pipeline failure
                # Nomic Embed Text has 768 dimensions
                fallback_dim = 768
                logger.warning(f"Using fallback zero vectors of dimension {fallback_dim}")
                embeddings.extend([[0.0] * fallback_dim for _ in batch])

        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embeds a single query.

        Args:
            text: Query string.

        Returns:
            The query embedding vector.
        """
        res = await self.aembed_documents([text])
        return res[0] if res else []


class ASCNVIDIAEmbeddings(OpenAIEmbeddings):
    """Custom wrapper for NVIDIA embeddings that dynamically sets input_type."""

    def __init__(self, *args, **kwargs):
        kwargs["check_embedding_ctx_length"] = False
        if "model_kwargs" not in kwargs:
            kwargs["model_kwargs"] = {}
        if "extra_body" not in kwargs["model_kwargs"]:
            kwargs["model_kwargs"]["extra_body"] = {}
        kwargs["model_kwargs"]["extra_body"]["truncate"] = "END"
        super().__init__(*args, **kwargs)

    def embed_documents(self, texts: list[str], chunk_size: int | None = None) -> list[list[float]]:
        if not self.model_kwargs:
            self.model_kwargs = {}
        if "extra_body" not in self.model_kwargs:
            self.model_kwargs["extra_body"] = {}
        self.model_kwargs["extra_body"]["input_type"] = "passage"
        self.model_kwargs["extra_body"]["truncate"] = "END"
        return super().embed_documents(texts, chunk_size=chunk_size)

    def embed_query(self, text: str) -> list[float]:
        if not self.model_kwargs:
            self.model_kwargs = {}
        if "extra_body" not in self.model_kwargs:
            self.model_kwargs["extra_body"] = {}
        self.model_kwargs["extra_body"]["input_type"] = "query"
        self.model_kwargs["extra_body"]["truncate"] = "END"
        return super().embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str], chunk_size: int | None = None) -> list[list[float]]:
        if not self.model_kwargs:
            self.model_kwargs = {}
        if "extra_body" not in self.model_kwargs:
            self.model_kwargs["extra_body"] = {}
        self.model_kwargs["extra_body"]["input_type"] = "passage"
        self.model_kwargs["extra_body"]["truncate"] = "END"
        return await super().aembed_documents(texts, chunk_size=chunk_size)

    async def aembed_query(self, text: str) -> list[float]:
        if not self.model_kwargs:
            self.model_kwargs = {}
        if "extra_body" not in self.model_kwargs:
            self.model_kwargs["extra_body"] = {}
        self.model_kwargs["extra_body"]["input_type"] = "query"
        self.model_kwargs["extra_body"]["truncate"] = "END"
        res = await super().aembed_documents([text])
        return res[0]



class ASCEmbedder(Embeddings):
    """Unified embedder supporting Ollama, NVIDIA NIM, and OpenAI.
    
    Acts as a wrapper around OllamaEmbedder, ASCNVIDIAEmbeddings, and OpenAIEmbeddings.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initializes the unified embedder.

        Args:
            provider: 'ollama', 'nvidia', or 'openai'. If None, loaded from LLM_PROVIDER env var.
            model: Model name. If None, loaded from env var specific to provider.
            base_url: Base URL for API. If None, loaded from env var specific to provider.
            api_key: API Key for cloud providers. If None, loaded from env var specific to provider.
        """
        import os
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "ollama")).lower()

        if self.provider == "nvidia":
            api_key = api_key or os.environ.get("NVIDIA_API_KEY")
            base_url = base_url or os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
            model = model or os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
            
            logger.info(f"Initializing NVIDIA ASCNVIDIAEmbeddings: model={model} at base_url={base_url}")
            self.underlying = ASCNVIDIAEmbeddings(
                model=model,
                openai_api_key=api_key,
                openai_api_base=base_url
            )
        elif self.provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            model = model or os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            
            logger.info(f"Initializing OpenAI OpenAIEmbeddings: model={model}")
            self.underlying = OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key,
                openai_api_base=base_url
            )
        else:
            # Fallback to OllamaEmbedder
            base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            model = model or os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            
            logger.info(f"Initializing local OllamaEmbedder: model={model} at base_url={base_url}")
            self.underlying = OllamaEmbedder(model=model, ollama_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.underlying.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.underlying.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.underlying, "aembed_documents"):
            return await self.underlying.aembed_documents(texts)
        else:
            # Fallback to executor for synchronous implementations
            import asyncio
            return await asyncio.get_event_loop().run_in_executor(
                None, self.underlying.embed_documents, texts
            )

    async def aembed_query(self, text: str) -> list[float]:
        if hasattr(self.underlying, "aembed_query"):
            return await self.underlying.aembed_query(text)
        else:
            import asyncio
            return await asyncio.get_event_loop().run_in_executor(
                None, self.underlying.embed_query, text
            )
