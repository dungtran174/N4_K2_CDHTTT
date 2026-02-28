"""
rag_graph.py
------------
Advanced RAG cho chatbot tuyển sinh PTIT, xây dựng bằng LangGraph.

Kiến trúc tổng thể:
  START → analyze_query
         ├─(cần làm rõ)─→ human_clarify ──→ analyze_query (lặp lại)
         ├─(general)────→ general_chat → END
         └─(admission)──→ retrieve_single (song song qua Send API)
                                 ↓ (gộp kết quả)
                          generate_answer → END

Tính năng:
  - Multi-query decomposition (chia câu hỏi phức tạp thành sub_queries)
  - Parallel retrieval dùng LangGraph Send API
  - ParentDocumentRetriever (child chunks → search, parent chunks → context)
  - Human-in-the-Loop (interrupt_before=["human_clarify"])
  - MemorySaver (lưu state theo thread_id, hỗ trợ multi-turn)
"""

import os
import logging
import pickle
from typing import Annotated, TypedDict
from pathlib import Path

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── Cấu hình ──────────────────────────────────────────────────────────────────

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

EMBEDDING_MODEL  = "dangvantuan/vietnamese-embedding"
PARENT_VECTOR_DIR = "./parent_vector_store"   # ChromaDB lưu child chunks
PARENT_DOCSTORE_DIR = "./parent_docstore"     # LocalFileStore lưu parent chunks
TOP_K     = 15   # Số docs trả về mỗi truy vấn (tăng để bắt nhiều section hơn)


# ── Custom State Reducer ──────────────────────────────────────────────────────────────────

def _docs_reducer(existing: list, update) -> list:
    """
    Custom reducer cho trường documents:
      - update = None  → reset về list rỗng (dùng khi bắt đầu retrieval mới)
      - update = list  → nối thêm vào list hiện có (dùng với Send API fan-out)
    """
    if update is None:
        return []
    return existing + update


# ── State ─────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """
    State chia sẻ toàn graph.
    Mỗi node trả về dict chứa các key cần cập nhật.
    """
    chat_history:        list             # Lịch sử hội thoại [{role, content}, ...]
    original_question:   str              # Câu hỏi gốc của người dùng
    sub_queries:         list[str]        # Danh sách câu hỏi con được tách ra
    documents:           Annotated[list, _docs_reducer]  # Tài liệu thu thập được
    #                    ↑ Annotated với _docs_reducer để:
    #                      1. Hỗ trợ Send API fan-out (nhiều retrieve_single gộp vào)
    #                      2. Cho phép reset bằng cách return {"documents": None}
    clarification_needed: bool            # Graph có cần hỏi lại người dùng không?
    intent:              str              # 'admission' hoặc 'general'
    generation:          str              # Câu trả lời cuối cùng


# ── Khởi tạo LLM & Retriever (singleton, lazy) ────────────────────────────────

_llm       = None
_retriever = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Khởi tạo ChatGoogleGenerativeAI (Gemini) (lazy singleton)."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GOOGLE_API_KEY trong file .env")
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key,
        )
    return _llm


def get_retriever() -> ParentDocumentRetriever:
    """
    Khởi tạo ParentDocumentRetriever (lazy singleton).

    ParentDocumentRetriever hoạt động 2 tầng:
      - Child chunks (nhỏ, 300 ký tự): lưu trong ChromaDB → tìm kiếm nhanh, chính xác
      - Parent chunks (lớn, 1500 ký tự): lưu trong LocalFileStore → cung cấp đủ ngữ cảnh

    Khi query: tìm child chunk → trả về parent chunk tương ứng → LLM đọc ngữ cảnh đầy đủ hơn.
    """
    global _retriever
    if _retriever is not None:
        return _retriever

    if not Path(PARENT_VECTOR_DIR).exists():
        raise FileNotFoundError(
            f"Không tìm thấy parent vector store tại '{PARENT_VECTOR_DIR}'.\n"
            "Hãy chạy:  python ingest.py --mode parent"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Child vectorstore: ChromaDB chứa các đoạn văn nhỏ
    child_vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=PARENT_VECTOR_DIR,
    )

    # Parent docstore: tải dữ liệu từ pickle vào InMemoryStore
    # (InMemoryStore là BaseStore chuẩn, tương thích 100% với ParentDocumentRetriever)
    pickle_path = Path(PARENT_DOCSTORE_DIR) / "docstore.pkl"
    in_memory_store = InMemoryStore()
    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        in_memory_store.mset(list(data.items()))
    else:
        raise FileNotFoundError(
            f"Không tìm thấy docstore tại '{pickle_path}'.\n"
            "Hãy chạy:  python ingest.py --mode parent"
        )

    # Child splitter: nhỏ để embedding đặc trưng hơn, retrieved chính xác hơn
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Parent splitter: lớn để LLM có đủ ngữ cảnh khi tổng hợp câu trả lời
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    _retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=in_memory_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": TOP_K},
    )
    return _retriever


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_history(chat_history: list, n_turns: int = 3) -> str:
    """Chuyển lịch sử hội thoại thành chuỗi text, lấy n_turns lượt gần nhất."""
    if not chat_history:
        return "Không có."
    lines = []
    for msg in chat_history[-(n_turns * 2):]:
        role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
        lines.append(f"{role}: {msg.get('content', '').strip()}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# CÁC NODES
# ════════════════════════════════════════════════════════════════════════════════

# ── Node 1: analyze_query ──────────────────────────────────────────────────────

def analyze_query(state: GraphState) -> dict:
    """
    Phân tích câu hỏi của người dùng kết hợp với lịch sử hội thoại.

    Kết quả:
      - Nếu câu hỏi quá mơ hồ              → clarification_needed = True
      - Nếu câu hỏi phức tạp/nhiều chủ đề  → sub_queries = [q1, q2, ...]
      - Nếu câu hỏi đơn giản               → sub_queries = [câu hỏi đầy đủ]

    Luôn reset documents = None để xóa kết quả cũ trước khi bắt đầu retrieval mới.
    """
    llm       = get_llm()
    question  = state["original_question"]
    history   = _format_history(state.get("chat_history", []))

    prompt = ChatPromptTemplate.from_template(
        """Bạn là bộ định tuyến AI (Router Agent) của hệ thống chatbot PTIT.
Nhiệm vụ của bạn là phân tích câu hỏi và trả về ĐÚNG định dạng JSON. KHÔNG ĐƯỢC sinh thêm bất kỳ chữ nào khác ngoài JSON.

Lịch sử hội thoại:
{history}

Câu hỏi hiện tại: "{question}"

QUY TẮC BẮT BUỘC:
1. Intent: "admission" (hỏi về điểm, học phí, ngành, xét tuyển...) hoặc "general" (chào hỏi).
2. Clarification_needed: NẾU người dùng hỏi về "điểm chuẩn", "học phí", "chỉ tiêu" MÀ KHÔNG HỀ CÓ tên ngành cụ thể (trong câu hỏi hoặc trong lịch sử), BẮT BUỘC gán là true.
3. sub_queries: Nếu clarification_needed=true, để rỗng []. Nếu false, viết lại câu hỏi rõ ràng.

--- CÁC VÍ DỤ MẪU ---
Ví dụ 1:
Lịch sử: Không có.
Câu hỏi: "Cho em hỏi điểm chuẩn năm ngoái ạ"
→ Thiếu tên ngành → clarification_needed = true
JSON: {{"intent": "admission", "clarification_needed": true, "sub_queries": []}}

Ví dụ 2:
Lịch sử: Không có.
Câu hỏi: "Ngành Công nghệ thông tin học phí bao nhiêu?"
→ Có tên ngành cụ thể → clarification_needed = false
JSON: {{"intent": "admission", "clarification_needed": false, "sub_queries": ["Học phí ngành Công nghệ thông tin Học viện PTIT là bao nhiêu?"]}}

Ví dụ 3:
Lịch sử: 
Người dùng: Điểm chuẩn ngành An toàn thông tin?
Trợ lý: 26 điểm.
Câu hỏi: "Thế học phí của ngành này đắt không?"
→ Có tên ngành từ lịch sử (An toàn thông tin) → clarification_needed = false
JSON: {{"intent": "admission", "clarification_needed": false, "sub_queries": ["Học phí ngành An toàn thông tin PTIT là bao nhiêu?"]}}

Ví dụ 4:
Lịch sử: Không có.
Câu hỏi: "Chào bạn, bạn là ai?"
→ Intent = general → clarification_needed = false, sub_queries = []
JSON: {{"intent": "general", "clarification_needed": false, "sub_queries": []}}

Ví dụ 5:
Lịch sử: Không có.
Câu hỏi: "Chỉ tiêu tuyển sinh bao nhiêu?"
→ Thiếu tên ngành → BẮT BUỘC clarification_needed = true
JSON: {{"intent": "admission", "clarification_needed": true, "sub_queries": []}}

Ví dụ 6:
Lịch sử: Không có.
Câu hỏi: "So sánh học phí IT với Marketing"
→ Có 2 tên ngành cụ thể → chia 2 sub_queries
JSON: {{"intent": "admission", "clarification_needed": false, "sub_queries": ["Học phí ngành Công nghệ thông tin PTIT", "Học phí ngành Marketing PTIT"]}}
-----------------------

KẾT QUẢ JSON CỦA BẠN:"""
    )

    chain = prompt | llm | JsonOutputParser()

    try:
        result               = chain.invoke({"history": history, "question": question})
        intent               = result.get("intent", "admission")
        if intent not in ("admission", "general"):
            intent = "admission"
        clarification_needed = bool(result.get("clarification_needed", False))
        sub_queries          = result.get("sub_queries") or ([question] if intent == "admission" else [])
    except Exception:
        # Fallback an toàn: coi là câu hỏi tuyển sinh đơn giản
        intent               = "admission"
        clarification_needed = False
        sub_queries          = [question]

    return {
        "intent":               intent,
        "sub_queries":          sub_queries,
        "documents":            None,   
        "clarification_needed": clarification_needed,# Reset: xóa tài liệu từ lần query trước
    }


# ── Node 2: general_chat ──────────────────────────────────────────────────────

def general_chat(state: GraphState) -> dict:
    """
    Xử lý các câu hỏi giao tiếp xã giao hoặc kiến thức chung ngoài lề tuyển sinh.

    Node này KHÔNG gọi Retriever. Nó dùng LLM trực tiếp kết hợp
    original_question và chat_history để sinh câu trả lời lịch sự, thân thiện.
    Kết quả lưu vào generation.
    """
    llm          = get_llm()
    question     = state["original_question"]
    chat_history = state.get("chat_history", [])
    history_text = _format_history(chat_history)

    prompt = ChatPromptTemplate.from_template(
        """Bạn là trợ lý AI thân thiện của Học viện Công nghệ Bưu chính Viễn thông (PTIT).

Lịch sử hội thoại:
{history}

Câu hỏi/tin nhắn của người dùng: {question}

Hãy trả lời một cách lịch sự, tự nhiên và thân thiện.
Nếu là lời chào → chào lại và giới thiệu ngắn gọn bản thân là trợ lý tư vấn tuyển sinh PTIT.
Nếu là câu hỏi kiến thức chung → trả lời trực tiếp, ngắn gọn.
Không cần trích dẫn nguồn.

Trả lời:"""
    )

    try:
        response = (prompt | llm | StrOutputParser()).invoke({
            "history":  history_text,
            "question": question,
        })
    except Exception as e:
        response = f"Xin lỗi, đã xảy ra lỗi: {e}"

    return {"generation": response}


# ── Node 3: human_clarify ─────────────────────────────────────────────────────

def human_clarify(state: GraphState) -> dict:
    """
    Human-in-the-Loop node.

    Node này sẽ bị INTERRUPT trước khi chạy (interrupt_before=["human_clarify"]).
    Flow khi bị interrupt:
      1. LangGraph tạm dừng và lưu state vào MemorySaver
      2. app.py phát hiện bị interrupt, hiện input box yêu cầu người dùng làm rõ
      3. Người dùng nhập câu hỏi rõ hơn
      4. app.py gọi graph.update_state() để cập nhật original_question
      5. app.py gọi graph.invoke(None, config) để tiếp tục

    Khi được resume, node này chạy tiếp và trả về flag để quay lại analyze_query.
    """
    return {
        "clarification_needed": False,  # Đã nhận được làm rõ, tắt flag
    }


# ── Node 4: retrieve_single ───────────────────────────────────────────────────

def retrieve_single(state: dict) -> dict:
    """
    Tìm kiếm tài liệu cho MỘT câu hỏi cụ thể.

    Node này KHÔNG nhận GraphState đầy đủ — nó nhận dict từ Send API:
      {"query": "câu hỏi cụ thể"}

    Kết quả được gộp vào GraphState.documents thông qua _docs_reducer.
    Mỗi doc được gắn thêm metadata "retrieved_by" để trace về sau.
    """
    query = state.get("query", "")
    if not query:
        return {"documents": []}

    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        for doc in docs:
            doc.metadata["retrieved_by"] = query
    except Exception:
        docs = []

    return {"documents": docs}


# ── Node 5: generate_answer ───────────────────────────────────────────────────

def generate_answer(state: GraphState) -> dict:
    """
    Tổng hợp tài liệu đã thu thập để tạo câu trả lời cuối cùng.

    - Nếu có tài liệu: trả lời dựa trên tài liệu
    - Nếu không có tài liệu: thông báo không tìm thấy
    """
    llm          = get_llm()
    question     = state["original_question"]
    documents    = state.get("documents", [])
    chat_history = state.get("chat_history", [])

    # Xây dựng context từ các tài liệu đã thu thập
    if documents:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", "")
            url   = doc.metadata.get("url", "")
            header = f"[Nguồn {i}]" + (f" {title}" if title else "") + (f" ({url})" if url else "")
            context_parts.append(f"{header}\n{doc.page_content}")
            
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "Không tìm thấy tài liệu liên quan trong hệ thống."

    history_text = _format_history(chat_history)

    prompt = ChatPromptTemplate.from_template(
        """Bạn là chuyên viên tư vấn tuyển sinh của Học viện Công nghệ Bưu chính Viễn thông (PTIT).
Phong cách: thân thiện, chính xác, trả lời đầy đủ nhưng súc tích.

Lịch sử hội thoại:
{history}

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Nguyên tắc trả lời:
1. Ưu tiên dùng thông tin từ [Nguồn X] — không cần trích dẫn.
2. Nếu không có tài liệu → thông báo lịch sự, gợi ý liên hệ trực tiếp PTIT.
3. KHÔNG bịa đặt điểm chuẩn, học phí, tên ngành nếu không có trong tài liệu.
4. Với câu hỏi chào hỏi/xã giao → trả lời tự nhiên, không cần trích dẫn.

Trả lời:"""
    )

    try:
        response = (prompt | llm | StrOutputParser()).invoke({
            "history":  history_text,
            "context":  context,
            "question": question,
        })
    except Exception as e:
        response = f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời: {e}"

    return {"generation": response}


# ════════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGES (ĐỊNH TUYẾN)
# ════════════════════════════════════════════════════════════════════════════════

def route_after_analyze(state: GraphState):
    """
    Sau analyze_query:
      - Cần làm rõ                  → human_clarify (sẽ bị interrupt trước khi chạy)
      - intent == 'general'         → general_chat (trả lời trực tiếp, không RAG)
      - intent == 'admission'       → fan-out song song tới retrieve_single
    """
    if state.get("clarification_needed", False):
        return "human_clarify"

    if state.get("intent", "admission") == "general":
        return "general_chat"

    # Send API fan-out: LangGraph chạy song song nhiều retrieve_single
    return [
        Send("retrieve_single", {"query": q})
        for q in state.get("sub_queries", [state["original_question"]])
    ]


# ════════════════════════════════════════════════════════════════════════════════
# XÂY DỰNG GRAPH
# ════════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Compile LangGraph với:
      - MemorySaver: lưu state theo thread_id (hỗ trợ multi-turn & HiL)
      - interrupt_before=["human_clarify"]: tạm dừng trước node yêu cầu làm rõ
    """
    graph = StateGraph(GraphState)

    # ── Thêm nodes ────────────────────────────────────────────────────────────
    graph.add_node("analyze_query",    analyze_query)
    graph.add_node("general_chat",     general_chat)
    graph.add_node("human_clarify",    human_clarify)
    graph.add_node("retrieve_single",  retrieve_single)
    graph.add_node("generate_answer",  generate_answer)

    # ── Edges cố định ─────────────────────────────────────────────────────────
    graph.add_edge(START,             "analyze_query")
    graph.add_edge("general_chat",    END)               # general_chat → kết thúc ngay
    graph.add_edge("human_clarify",   "analyze_query")   # Sau khi làm rõ → phân tích lại
    graph.add_edge("retrieve_single", "generate_answer")  # Fan-in: mọi retrieve_single đổ về generate

    # ── Conditional edges ─────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analyze,
        ["human_clarify", "general_chat", "retrieve_single"],
    )

    graph.add_edge("generate_answer", END)

    # ── Compile với MemorySaver và HiL interrupt ───────────────────────────────
    memory = MemorySaver()
    compiled = graph.compile(
        checkpointer=memory,
        interrupt_before=["human_clarify"],  # Tạm dừng để người dùng nhập làm rõ
    )
    return compiled


# ── Singleton compiled graph ──────────────────────────────────────────────────

_graph = None

def get_graph():
    """Trả về compiled graph duy nhất (lazy init)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ════════════════════════════════════════════════════════════════════════════════
# HÀM TIỆN ÍCH CHO app.py
# ════════════════════════════════════════════════════════════════════════════════

def _extract_sources(documents: list) -> list[dict]:
    """Trích xuất danh sách nguồn tham khảo từ documents để hiển thị UI."""
    sources   = []
    seen_urls = set()
    for doc in documents:
        meta = doc.metadata
        url  = meta.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append({
            "url":       url,
            "title":     meta.get("title", ""),
            "source":    meta.get("source", ""),
            "file_type": meta.get("file_type", ""),
            "filename":  meta.get("filename", ""),
        })
    return sources


def run_graph(
    question:    str,
    chat_history: list,
    thread_id:   str = "default",
) -> dict:
    """
    Chạy agentic RAG graph từ đầu với câu hỏi mới.

    Parameters
    ----------
    question     : Câu hỏi của người dùng
    chat_history : Lịch sử hội thoại [{role, content}, ...]
    thread_id    : ID phiên (mỗi người dùng/tab nên dùng thread_id riêng)

    Returns
    -------
    dict:
        answer               : str   — câu trả lời (rỗng nếu bị interrupt)
        sources              : list  — danh sách nguồn tham khảo
        clarification_needed : bool  — True nếu graph tạm dừng chờ người dùng
        thread_id            : str   — dùng để resume sau khi làm rõ
    """
    graph  = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "chat_history":        chat_history,
        "original_question":   question,
        "sub_queries":         [],
        "documents":           None,   # None → _docs_reducer → []
        "clarification_needed": False,
        "intent":              "admission",
        "generation":          "",
    }

    result = graph.invoke(initial_state, config=config)

    # Kiểm tra graph có bị interrupt không (có next node đang chờ)
    graph_state         = graph.get_state(config)
    clarification_needed = bool(graph_state.next)

    return {
        "answer":               result.get("generation", ""),
        "sources":              _extract_sources(result.get("documents") or []),
        "clarification_needed": clarification_needed,
        "thread_id":            thread_id,
    }


def resume_after_clarification(
    clarification: str,
    thread_id:     str = "default",
) -> dict:
    """
    Resume graph sau khi người dùng cung cấp làm rõ câu hỏi.

    Gọi sau khi run_graph() trả về clarification_needed = True.

    Parameters
    ----------
    clarification : Câu hỏi đã được làm rõ từ người dùng
    thread_id     : Cùng thread_id với lần run_graph() trước đó

    Returns
    -------
    Cùng format với run_graph()
    """
    graph  = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Cập nhật state: thay original_question bằng câu đã làm rõ
    graph.update_state(
        config,
        {"original_question": clarification, "clarification_needed": False},
    )

    # Tiếp tục từ điểm interrupt (invoke(None) = resume)
    result = graph.invoke(None, config=config)

    graph_state          = graph.get_state(config)
    clarification_needed = bool(graph_state.next)

    return {
        "answer":               result.get("generation", ""),
        "sources":              _extract_sources(result.get("documents") or []),
        "clarification_needed": clarification_needed,
        "thread_id":            thread_id,
    }


# ── Test nhanh ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid

    print("=== Test Advanced RAG Graph ===\n")
    tid = str(uuid.uuid4())

    result = run_graph(
        question="PTIT tuyển sinh những ngành nào năm 2026?",
        chat_history=[],
        thread_id=tid,
    )

    if result["clarification_needed"]:
        print("Bot cần làm rõ — nhập câu hỏi rõ hơn để tiếp tục.")
    else:
        print(f"Trả lời:\n{result['answer']}\n")
        print("Nguồn tham khảo:")
        for i, src in enumerate(result["sources"], 1):
            label = src.get("title") or src.get("filename") or src.get("source")
            url   = src.get("url", "")
            print(f"  [{i}] {label}" + (f"\n       {url}" if url else ""))
