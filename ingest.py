"""
ingest.py
---------
Đọc toàn bộ dữ liệu từ thư mục data/crawled_data/, xử lý .md và .pdf,
chunk văn bản, nhúng và lưu vào ChromaDB tại ./vector_store/.

Mỗi lần chạy sẽ XÓA vector store cũ và nạp lại từ đầu để đảm bảo
dữ liệu luôn đồng bộ với crawled_data (không bị trùng lặp hay lỗi thời).

Hai chế độ (--mode):
  standard (mặc định): Chroma.from_documents → dùng cho rag_pipeline.py (legacy)
  parent             : ParentDocumentRetriever → dùng cho rag_graph.py (agentic)

Usage:
  python ingest.py                # chế độ standard
  python ingest.py --mode parent  # chế độ parent (cho rag_graph.py)
  python ingest.py --mode all     # xây dựng cả hai
"""

import argparse
import os
import json
import logging
import pickle
import shutil
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Cấu hình ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/crawled_data")
VECTOR_STORE_DIR     = "./vector_store"        # Standard ChromaDB (rag_pipeline.py)
PARENT_VECTOR_DIR    = "./parent_vector_store" # Child chunks ChromaDB (rag_graph.py)
PARENT_DOCSTORE_DIR  = "./parent_docstore"     # Parent chunks LocalFileStore (rag_graph.py)
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
# Model nhỏ (~500MB), hỗ trợ đa ngôn ngữ bao gồm tiếng Việt
EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding"

# Dùng HuggingFace mirror để tải model nhanh hơn ở Việt Nam
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ── Hàm trích xuất PDF ─────────────────────────────────────────────────────

def is_scanned_pdf(pdf_path: str, text_threshold: int = 50) -> bool:
    """Kiểm tra PDF có phải dạng scan ảnh hay không.
    Nếu tổng ký tự trích xuất < text_threshold thì coi là PDF scan.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_text = "".join(
                (page.extract_text() or "") for page in pdf.pages
            )
        return len(total_text.strip()) < text_threshold
    except Exception:
        return True


def extract_text_from_text_pdf(pdf_path: str) -> str:
    """Trích xuất text từ PDF dạng text bằng pdfplumber."""
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
    except Exception as e:
        logger.warning(f"pdfplumber lỗi với {pdf_path}: {e}")
    return "\n".join(pages_text)


def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    """Trích xuất text từ PDF scan bằng pdf2image + pytesseract (tiếng Việt)."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        logger.error("Thiếu thư viện pdf2image hoặc pytesseract. Cài bằng: pip install pdf2image pytesseract")
        return ""

    pages_text = []
    try:
        images = convert_from_path(pdf_path, dpi=200)
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang="vie")
            if text.strip():
                pages_text.append(text)
        logger.info(f"OCR xong {len(images)} trang: {Path(pdf_path).name}")
    except Exception as e:
        logger.warning(f"OCR lỗi với {pdf_path}: {e}")
    return "\n".join(pages_text)


def extract_pdf_text(pdf_path: str) -> str:
    """Tự động chọn phương pháp đọc PDF phù hợp."""
    if is_scanned_pdf(pdf_path):
        logger.info(f"PDF scan detected → OCR: {Path(pdf_path).name}")
        return extract_text_from_scanned_pdf(pdf_path)
    else:
        logger.info(f"PDF text detected → pdfplumber: {Path(pdf_path).name}")
        return extract_text_from_text_pdf(pdf_path)


# ── Hàm load dữ liệu ─────────────────────────────────────────────────────────

def load_md_document(md_path: Path, metadata: dict) -> list[Document]:
    """
    Đọc file .md, tách thành các section theo header Markdown.
    Mỗi Document chứa: [Tiêu đề trang > Tên mục]\n\nNội dung mục
    để embedding nắm được context ngữ nghĩa đầy đủ (tên chương trình + tên mục).
    """
    try:
        content = md_path.read_text(encoding="utf-8").strip()
        if not content:
            return []

        page_title = metadata.get("title", "").strip()
        base_meta = {
            "source": str(md_path),
            "url": metadata.get("url", ""),
            "title": page_title,
            "crawl_date": metadata.get("crawl_date", ""),
            "file_type": "markdown",
        }

        # Tách theo header để mỗi section có header context riêng
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#",  "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,  # Giữ header trong content
        )
        sections = splitter.split_text(content)

        docs: list[Document] = []
        for sec in sections:
            sec_content = sec.page_content.strip()
            if not sec_content:
                continue

            # Tạo breadcrumb từ các header metadata
            header_parts = []
            if page_title:
                header_parts.append(page_title)
            for key in ("h1", "h2", "h3"):
                val = sec.metadata.get(key, "").strip()
                if val and val not in header_parts:
                    header_parts.append(val)

            # Prefix breadcrumb vào content để embedding hiểu ngữ cảnh
            if header_parts:
                breadcrumb = " > ".join(header_parts)
                enriched_content = f"[{breadcrumb}]\n\n{sec_content}"
            else:
                enriched_content = sec_content

            meta = {**base_meta, **sec.metadata}
            docs.append(Document(page_content=enriched_content, metadata=meta))

        # Fallback: nếu split cho 0 section (file không có header), dùng toàn bộ
        if not docs:
            enriched = f"[{page_title}]\n\n{content}" if page_title else content
            docs.append(Document(page_content=enriched, metadata=base_meta))

        return docs
    except Exception as e:
        logger.warning(f"Lỗi đọc MD {md_path}: {e}")
        return []


def load_pdf_documents(attachments_dir: Path, metadata: dict) -> list[Document]:
    """Đọc tất cả file .pdf trong thư mục attachments/ và tạo Documents."""
    documents = []
    if not attachments_dir.exists():
        return documents

    pdf_files = list(attachments_dir.glob("*.pdf"))
    for pdf_path in pdf_files:
        text = extract_pdf_text(str(pdf_path))
        if not text.strip():
            logger.warning(f"Không trích xuất được text từ: {pdf_path.name}")
            continue

        meta = {
            "source": str(pdf_path),
            "url": metadata.get("url", ""),
            "title": metadata.get("title", ""),
            "crawl_date": metadata.get("crawl_date", ""),
            "file_type": "pdf",
            "filename": pdf_path.name,
        }
        documents.append(Document(page_content=text, metadata=meta))

    return documents


def load_all_documents(data_dir: Path) -> list[Document]:
    """Duyệt toàn bộ data_dir và thu thập Documents từ .md và .pdf."""
    all_docs: list[Document] = []
    page_dirs = [p for p in data_dir.rglob("metadata.json")]

    logger.info(f"Tìm thấy {len(page_dirs)} thư mục trang để xử lý.")

    for meta_path in page_dirs:
        folder = meta_path.parent

        # Đọc metadata
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Lỗi đọc metadata {meta_path}: {e}")
            continue

        # Đọc file .md (trả về list[Document] theo từng section)
        md_path = folder / "content.md"
        if md_path.exists():
            md_docs = load_md_document(md_path, metadata)
            all_docs.extend(md_docs)

        # Đọc file .pdf trong attachments/
        attachments_dir = folder / "attachments"
        pdf_docs = load_pdf_documents(attachments_dir, metadata)
        all_docs.extend(pdf_docs)

    logger.info(f"Tổng số Documents thu thập được: {len(all_docs)}")
    return all_docs


# ── Hàm chunking ─────────────────────────────────────────────────────────────

def split_documents(documents: list[Document]) -> list[Document]:
    """Cắt nhỏ Documents bằng RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Sau chunking: {len(chunks)} chunks (từ {len(documents)} documents)")
    return chunks


# ── Hàm tạo vector store ──────────────────────────────────────────────────────

def build_vector_store(chunks: list[Document]) -> Chroma:
    """Nhúng các chunks và lưu vào ChromaDB."""
    logger.info(f"Khởi tạo embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"Đang nhúng và lưu {len(chunks)} chunks vào ChromaDB tại '{VECTOR_STORE_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )
    logger.info("Hoàn tất! Vector store đã được lưu.")
    return vectorstore


# ── Main ──────────────────────────────────────────────────────────────────────

# ── Hàm tạo Parent Vector Store (dùng cho rag_graph.py) ──────────────────────

def build_parent_stores(documents: list[Document]) -> None:
    """
    Xây dựng ParentDocumentRetriever stores cho rag_graph.py.

    Cách hoạt động:
      - Child splitter (300 ký tự): cắt nhỏ để embedding chính xác hơn → lưu ChromaDB
      - Parent splitter (1500 ký tự): cắt lớn để cung cấp đủ ngữ cảnh → lưu LocalFileStore
      - Khi truy vấn: tìm child chunks → trả về parent chunks tương ứng

    Lý do dùng 2 tầng:
      - Embedding nhỏ: tìm kiếm chính xác hơn (ít nhiễu)
      - Context lớn: LLM có đủ thông tin để trả lời đầy đủ
    """
    logger.info("=== Xây dựng Parent Vector Store cho rag_graph.py ===")

    # Xóa stores cũ
    for path, name in [(PARENT_VECTOR_DIR, "parent vector store"),
                       (PARENT_DOCSTORE_DIR, "parent docstore")]:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
            logger.info(f"Đã xóa {name} cũ tại '{path}'")

    logger.info(f"Khởi tạo embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Child vectorstore: ChromaDB lưu các đoạn văn nhỏ (để tìm kiếm)
    child_vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=PARENT_VECTOR_DIR,
    )

    # Parent docstore: InMemoryStore (tương thích chuẩn với ParentDocumentRetriever)
    # Sẽ được lưu xuống đĩa sau khi add_documents bằng pickle
    in_memory_store = InMemoryStore()

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=in_memory_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    logger.info(f"Đang nạp {len(documents)} documents vào ParentDocumentRetriever...")
    retriever.add_documents(documents)

    # Persist docstore data to disk bằng pickle
    pickle_path = Path(PARENT_DOCSTORE_DIR) / "docstore.pkl"
    Path(PARENT_DOCSTORE_DIR).mkdir(parents=True, exist_ok=True)
    keys  = list(in_memory_store.yield_keys())
    vals  = in_memory_store.mget(keys)
    with open(pickle_path, "wb") as f:
        pickle.dump(dict(zip(keys, vals)), f)
    logger.info(
        f"Hoàn tất! Parent vector store → '{PARENT_VECTOR_DIR}', "
        f"Parent docstore → '{pickle_path}' ({len(keys)} entries)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PTIT RAG Ingest Tool")
    parser.add_argument(
        "--mode",
        choices=["standard", "parent", "all"],
        default="standard",
        help=(
            "standard: xây dựng ChromaDB thông thường (rag_pipeline.py) | "
            "parent: xây dựng ParentDocumentRetriever stores (rag_graph.py) | "
            "all: xây dựng cả hai"
        ),
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        logger.error(f"Không tìm thấy thư mục dữ liệu: {DATA_DIR.resolve()}")
        return

    # Load toàn bộ documents (dùng chung cho cả hai mode)
    documents = load_all_documents(DATA_DIR)
    if not documents:
        logger.error("Không có document nào được tải. Kiểm tra lại thư mục data/.")
        return

    if args.mode in ("standard", "all"):
        logger.info("=== Xây dựng Standard Vector Store ===")
        # Xóa vector store cũ để tránh trùng lặp / dữ liệu lỗi thời
        vs_path = Path(VECTOR_STORE_DIR)
        if vs_path.exists():
            shutil.rmtree(vs_path)
            logger.info(f"Đã xóa vector store cũ tại '{VECTOR_STORE_DIR}'")
        chunks = split_documents(documents)
        build_vector_store(chunks)

    if args.mode in ("parent", "all"):
        build_parent_stores(documents)


if __name__ == "__main__":
    main()
