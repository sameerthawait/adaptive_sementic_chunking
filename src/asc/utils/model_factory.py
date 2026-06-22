import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

def get_llm_from_env() -> Any:
    """
    Returns the configured LLM class for RAG generation, grading, and verification
    based on the LLM_PROVIDER environment variable.
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    
    if provider == "nvidia":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("NVIDIA_API_KEY")
        base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        model = os.environ.get("NVIDIA_GENERATION_MODEL", "meta/llama-3.3-70b-instruct")
        
        logger.info(f"Initializing NVIDIA ChatOpenAI model: {model} at {base_url}")
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("OPENAI_GENERATION_MODEL", "gpt-4o")
        
        logger.info(f"Initializing OpenAI ChatOpenAI model: {model}")
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0
        )
    else:
        # Default Ollama
        from langchain_ollama import OllamaLLM
        model = os.environ.get("OLLAMA_PERPLEXITY_MODEL", "llama3.2:3b")
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
        logger.info(f"Initializing local Ollama model: {model} at {ollama_url}")
        return OllamaLLM(model=model, base_url=ollama_url)
