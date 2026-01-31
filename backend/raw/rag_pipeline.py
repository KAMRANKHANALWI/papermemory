# rag_pipeline.py
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

class SimpleRAGPipeline:
    def __init__(self, config):
        self.config = config
        self.llm = self._init_llm()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )

        self.chroma_client = chromadb.PersistentClient(
            path=config.DB_PATH
        )

        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.DB_PATH,
        )

        print("✅ RAG pipeline ready")

    def _init_llm(self):
        if self.config.USE_LOCAL_LLM:
            print(f"🔧 Using Ollama → {self.config.OLLAMA_MODEL}")
            return ChatOllama(
                model=self.config.OLLAMA_MODEL,
                temperature=self.config.TEMPERATURE,
            )

        print(f"🔧 Using Gemini → {self.config.GEMINI_MODEL}")
        return ChatGoogleGenerativeAI(
            model=self.config.GEMINI_MODEL,
            temperature=self.config.TEMPERATURE,
        )

    def retrieve(self, query: str):
        results = self.vectorstore.similarity_search(query, k=self.config.TOP_K)
        return [doc.page_content for doc in results]

    def generate(self, query: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(contexts)

        prompt = f"""
You are a helpful assistant.
Answer ONLY from the context below.
If insufficient information, say so.

Context:
{context_block}

Question:
{query}
"""

        return self.llm.invoke(prompt).content
