"""
rag_pipeline.py
---------------
Load ChromaDB, khởi tạo LLM và tạo RAG chain để trả lời câu hỏi tuyển sinh.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

# ── Cấu hình ────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = "./vector_store"
# Model nhỏ (~500MB), hỗ trợ đa ngôn ngữ bao gồm tiếng Việt
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 4

# Dùng HuggingFace mirror để tải model nhanh hơn ở Việt Nam
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Chọn LLM backend: "openai", "google" hoặc "groq"
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")


# ── Khởi tạo LLM ─────────────────────────────────────────────────────────────

def _build_llm():
    """Khởi tạo LLM dựa trên biến môi trường LLM_BACKEND."""
    if LLM_BACKEND == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GROQ_API_KEY trong file .env")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=api_key,
        )
    elif LLM_BACKEND == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GOOGLE_API_KEY trong file .env")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            google_api_key=api_key,
        )
    else:  # openai
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Thiếu OPENAI_API_KEY trong file .env")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=api_key,
        )


# ── Khởi tạo Embeddings & Vector Store ───────────────────────────────────────

def _build_retriever():
    """Load ChromaDB đã lưu và trả về retriever."""
    if not Path(VECTOR_STORE_DIR).exists():
        raise FileNotFoundError(
            f"Không tìm thấy vector store tại '{VECTOR_STORE_DIR}'. "
            "Hãy chạy ingest.py trước."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return retriever


# ── Prompt Template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên viên tư vấn tuyển sinh thông minh của Học viện Công nghệ Bưu chính Viễn thông (PTIT).

Hãy trả lời câu hỏi theo nguyên tắc sau:

1. Nếu ngữ cảnh (Context) bên dưới có thông tin liên quan → Ưu tiên trả lời DỰA TRÊN ngữ cảnh đó, trích dẫn chính xác.
2. Nếu ngữ cảnh KHÔNG có hoặc THIẾU thông tin → Dùng kiến thức chung của bạn để hỗ trợ, nhưng hãy ghi rõ: "(Thông tin từ kiến thức chung, không có trong dữ liệu PTIT)"
3. Với câu hỏi chào hỏi, xã giao → Trả lời tự nhiên, thân thiện.
4. Không bịa đặt số liệu, tên ngành, điểm chuẩn cụ thể nếu không có trong ngữ cảnh.

Context:
{context}

Câu hỏi: {question}

Trả lời:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT,
)


# ── Hàm hỗ trợ ───────────────────────────────────────────────────────────────

def _format_docs(docs: list) -> str:
    """Ghép nội dung các chunks thành một chuỗi context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _extract_sources(docs: list) -> list[dict]:
    """Trích xuất metadata nguồn từ các chunks trả về."""
    sources = []
    for doc in docs:
        meta = doc.metadata
        sources.append({
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "file_type": meta.get("file_type", ""),
            "filename": meta.get("filename", ""),
        })
    return sources


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """Pipeline RAG tuyển sinh: retrieval → prompt → LLM."""

    def __init__(self):
        self.retriever = _build_retriever()
        self.llm = _build_llm()
        self._chain = self._build_chain()

    def _build_chain(self):
        """Xây dựng LCEL chain: retrieve → format → prompt → LLM."""
        rag_chain_with_sources = RunnableParallel(
            {
                "docs": self.retriever,
                "question": RunnablePassthrough(),
            }
        ).assign(
            context=lambda x: _format_docs(x["docs"]),
        ).assign(
            answer=(
                (lambda x: {"context": x["context"], "question": x["question"]})
                | PROMPT
                | self.llm
                | StrOutputParser()
            )
        )
        return rag_chain_with_sources

    def get_answer(self, query: str, max_retries: int = 3) -> dict:
        """
        Nhận câu hỏi, tìm kiếm top-k chunks liên quan và trả về câu trả lời.
        Tự động retry khi bị rate limit (429).

        Parameters
        ----------
        query : str
            Câu hỏi của người dùng.
        max_retries : int
            Số lần thử lại tối đa khi bị rate limit.

        Returns
        -------
        dict với các key:
            - answer  : str  — câu trả lời của LLM
            - sources : list — danh sách metadata của chunks nguồn
            - question: str  — câu hỏi gốc
        """
        for attempt in range(max_retries):
            try:
                result = self._chain.invoke(query)
                return {
                    "question": query,
                    "answer": result["answer"],
                    "sources": _extract_sources(result["docs"]),
                }
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = 10 * (attempt + 1)
                    if attempt < max_retries - 1:
                        time.sleep(wait)
                        continue
                raise
        raise RuntimeError("Vượt quá số lần thử lại khi gọi LLM.")


# ── Singleton (dùng chung trong app.py) ──────────────────────────────────────

_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Trả về instance RAGPipeline duy nhất (lazy init)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def get_answer(query: str) -> dict:
    """
    Hàm tiện ích: khởi tạo pipeline (nếu chưa) và trả về câu trả lời.

    Parameters
    ----------
    query : str
        Câu hỏi của người dùng.

    Returns
    -------
    dict: {"question": ..., "answer": ..., "sources": [...]}
    """
    return get_rag_pipeline().get_answer(query)


# ── Test nhanh ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_query = "Học viện PTIT tuyển sinh những ngành nào?"
    print(f"Câu hỏi: {test_query}\n")
    result = get_answer(test_query)
    print(f"Trả lời:\n{result['answer']}\n")
    print("Nguồn tham khảo:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src.get('title') or src.get('filename') or src.get('source')}")
        if src.get("url"):
            print(f"       URL: {src['url']}")
