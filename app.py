"""
app.py
------
Giao diện chatbot tư vấn tuyển sinh PTIT sử dụng Streamlit.
Tích hợp Agentic RAG (rag_graph.py) với hỗ trợ:
  - Conversation memory (lịch sử hội thoại đa lượt)
  - Human-in-the-Loop (hỏi lại khi câu hỏi mơ hồ)
  - Hiển thị nguồn tham khảo

Chạy: streamlit run app.py
"""

import uuid
import streamlit as st
from rag_graph import run_graph, resume_after_clarification

#  Cấu hình trang 
st.set_page_config(
    page_title="Chatbot Tư Vấn Tuyển Sinh PTIT",
    page_icon="",
    layout="centered",
)

#  CSS tùy chỉnh 
st.markdown(
    """
    <style>
        .source-box {
            background-color: #f0f2f6;
            border-left: 3px solid #4a90d9;
            border-radius: 4px;
            padding: 8px 12px;
            margin-top: 8px;
            font-size: 0.78em;
            color: #444;
            line-height: 1.6;
        }
        .source-box a { color: #1a73e8; text-decoration: none; }
        .source-box a:hover { text-decoration: underline; }
        .source-header { font-weight: 600; color: #333; margin-bottom: 4px; }
        .clarify-box {
            background-color: #fff8e1;
            border-left: 3px solid #f9a825;
            border-radius: 4px;
            padding: 10px 14px;
            margin-bottom: 12px;
            font-size: 0.9em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

#  Tiêu đề 
st.title(" Chatbot Tư Vấn Tuyển Sinh PTIT")
st.caption("Học viện Công nghệ Bưu chính Viễn thông  Hỏi bất cứ điều gì về tuyển sinh!")
st.divider()

#  Khởi tạo session state 
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Xin chào! Tôi là chuyên viên tư vấn tuyển sinh PTIT. "
                "Bạn muốn hỏi về ngành học, điểm chuẩn hay thủ tục nhập học?"
            ),
            "sources": [],
        }
    ]

# thread_id duy nhất cho mỗi phiên Streamlit (MemorySaver dùng để lưu state)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Flag: đang chờ người dùng làm rõ câu hỏi (Human-in-the-Loop)
if "waiting_clarification" not in st.session_state:
    st.session_state.waiting_clarification = False


#  Helper: hiển thị nguồn tham khảo 
def render_sources(sources: list[dict]) -> None:
    """Hiển thị danh sách nguồn tham khảo dạng chữ nhỏ bên dưới câu trả lời."""
    if not sources:
        return

    seen = set()
    unique_sources = []
    for src in sources:
        key = (src.get("title", ""), src.get("url", ""), src.get("filename", ""))
        if key not in seen:
            seen.add(key)
            unique_sources.append(src)

    lines = ["<div class='source-box'><div class='source-header'> Nguồn tham khảo</div>"]
    for i, src in enumerate(unique_sources, 1):
        title     = src.get("title") or src.get("filename") or src.get("source") or "Không rõ"
        url       = src.get("url", "")
        file_type = src.get("file_type", "")
        icon      = "" if file_type == "pdf" else ""
        label     = f"{icon} {title}"

        if url:
            lines.append(f"&nbsp;&nbsp;{i}. <a href='{url}' target='_blank'>{label}</a>")
        else:
            lines.append(f"&nbsp;&nbsp;{i}. {label}")

        if src.get("filename"):
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;<em>File: {src['filename']}</em>")

    lines.append("</div>")
    st.markdown("\n".join(lines), unsafe_allow_html=True)


#  Hiển thị lịch sử hội thoại 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])


#  Xử lý Human-in-the-Loop: hiển thị input làm rõ 
if st.session_state.waiting_clarification:
    st.markdown(
        "<div class='clarify-box'> <b>Câu hỏi của bạn chưa đủ rõ.</b> "
        "Hãy cung cấp thêm thông tin để tôi có thể hỗ trợ chính xác hơn.</div>",
        unsafe_allow_html=True,
    )

    clarification = st.chat_input("Làm rõ câu hỏi của bạn tại đây...")
    if clarification:
        st.session_state.messages.append(
            {"role": "user", "content": clarification, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(clarification)

        with st.chat_message("assistant"):
            with st.spinner("Đang xử lý..."):
                try:
                    result = resume_after_clarification(
                        clarification=clarification,
                        thread_id=st.session_state.thread_id,
                    )
                    answer  = result["answer"]
                    sources = result.get("sources", [])
                    st.session_state.waiting_clarification = result["clarification_needed"]
                except Exception as e:
                    answer  = f" Đã xảy ra lỗi: {e}"
                    sources = []
                    st.session_state.waiting_clarification = False

            st.markdown(answer)
            render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()

#  Xử lý input người dùng thông thường 
else:
    if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Chuẩn bị lịch sử hội thoại để truyền vào graph
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("assistant"):
            with st.spinner("Đang phân tích và tìm kiếm thông tin..."):
                try:
                    result = run_graph(
                        question=prompt,
                        chat_history=chat_history,
                        thread_id=st.session_state.thread_id,
                    )
                    answer  = result["answer"]
                    sources = result.get("sources", [])
                    st.session_state.waiting_clarification = result["clarification_needed"]
                except Exception as e:
                    answer  = f" Đã xảy ra lỗi khi xử lý câu hỏi: {e}"
                    sources = []
                    st.session_state.waiting_clarification = False

            if st.session_state.waiting_clarification:
                answer = (
                    "Câu hỏi của bạn chưa đủ rõ để tôi tìm kiếm thông tin chính xác. "
                    "Bạn có thể mô tả cụ thể hơn bên dưới không?"
                )
                sources = []

            st.markdown(answer)
            render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

        if st.session_state.waiting_clarification:
            st.rerun()


#  Sidebar 
with st.sidebar:
    st.header("ℹ Thông tin")
    st.markdown(
        """
        **Chatbot Tư Vấn Tuyển Sinh PTIT**

        Sử dụng kiến trúc **Agentic RAG** (LangGraph):
        -  Phân tách câu hỏi phức tạp thành sub-queries
        -  Tìm kiếm song song (Send API)
        -  Lọc tài liệu bằng LLM grader
        -  Tự sửa nếu không tìm thấy kết quả (tối đa 3 lần)
        -  Ghi nhớ lịch sử hội thoại (MemorySaver)

        ---
        **Gợi ý câu hỏi:**
        - Các ngành tuyển sinh năm 2026 là gì?
        - Điểm chuẩn ngành Công nghệ thông tin?
        - So sánh học phí ngành IT và Marketing
        - PTIT có những loại học bổng nào?
        - Chương trình đào tạo Fintech gồm gì?
        ---
        """
    )

    if st.button(" Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Xin chào! Tôi là chuyên viên tư vấn tuyển sinh PTIT. "
                    "Bạn muốn hỏi về ngành học, điểm chuẩn hay thủ tục nhập học?"
                ),
                "sources": [],
            }
        ]
        # Tạo thread_id mới để reset conversation memory trong LangGraph
        st.session_state.thread_id            = str(uuid.uuid4())
        st.session_state.waiting_clarification = False
        st.rerun()

    st.caption("Powered by LangGraph  LangChain  ChromaDB  Streamlit")
