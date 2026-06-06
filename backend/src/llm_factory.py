import logging
import os

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class LLMFactory:
    @staticmethod
    def create():
        """
        LLM selection priority:

        1. USE_LOCAL_LLM=true
           -> Ollama

        2. DEFAULT_MODEL_PROVIDER=gemini
           + GOOGLE_API_KEY available
           -> Gemini

        3. GROQ_API_KEY available
           -> Groq

        Raises:
            ValueError if no valid provider configuration exists.
        """

        use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        default_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "gemini")

        if use_local_llm:
            return LLMFactory._ollama()

        if default_provider == "gemini" and os.getenv("GOOGLE_API_KEY"):
            return LLMFactory._gemini()

        if os.getenv("GROQ_API_KEY"):
            return LLMFactory._groq()

        raise ValueError(
            "No valid configuration found. Configure one of:\n"
            "- USE_LOCAL_LLM=true (Ollama)\n"
            "- GOOGLE_API_KEY (Gemini)\n"
            "- GROQ_API_KEY (Groq)"
        )

    @staticmethod
    def _ollama():
        ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        ollama_model = os.getenv(
            "OLLAMA_MODEL",
            "llama3.1:latest",
        )

        llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.1,
        )

        # Verify Ollama connection during startup
        llm.invoke("Hello")

        logger.info(
            f"Using Ollama model: {ollama_model}"
        )

        return llm

    @staticmethod
    def _gemini():
        model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )

        logger.info(
            f"Using Gemini model: {model}"
        )

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    @staticmethod
    def _groq():
        model = os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant",
        )

        logger.info(
            f"Using Groq model: {model}"
        )

        return ChatGroq(
            model=model,
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY"),
        )