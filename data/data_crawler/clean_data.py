"""
clean_data.py
Làm sạch crawled_data sau khi crawl:
  1. Xóa trang archive (URL /YYYY/MM/ hoặc /YYYY/MM/DD/)
  2. Xóa trang nội dung quá ngắn (< 100 ký tự)
  3. Làm sạch noise trong content.md

Usage: python clean_data.py
"""
import json
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "data" / "crawled_data"

ARCHIVE_RE = re.compile(r"/\d{4}/\d{2}(/\d{2})?/?$")
NOISE_RE = [
    re.compile(r"^-\s*$", re.MULTILINE),
    re.compile(r"^-\s*(Chi tiết bài viết|Chia sẻ:)\s*$", re.MULTILINE),
    re.compile(r"\b(admintuyen\w*|admindaotao|admintuyensinh|ptitadmin)\b", re.IGNORECASE),
    re.compile(r"^\d{2}/\d{2}/\d{4}\s*$", re.MULTILINE),
]
MIN_CONTENT_LEN = 100


def clean_content(text: str) -> str:
    for pat in NOISE_RE:
        text = pat.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main():
    removed_archive = removed_short = cleaned = 0

    for meta_path in list(BASE.rglob("metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        url = meta.get("url", "")
        folder = meta_path.parent
        content_path = folder / "content.md"

        # 1. Xóa archive pages
        if ARCHIVE_RE.search(url):
            shutil.rmtree(folder, ignore_errors=True)
            removed_archive += 1
            continue

        # 2. Xóa trang nội dung quá ngắn
        if not content_path.exists():
            shutil.rmtree(folder, ignore_errors=True)
            removed_short += 1
            continue

        text = content_path.read_text(encoding="utf-8")
        if len(text.strip()) < MIN_CONTENT_LEN:
            shutil.rmtree(folder, ignore_errors=True)
            removed_short += 1
            continue

        # 3. Làm sạch noise
        cleaned_text = clean_content(text)
        if cleaned_text != text:
            content_path.write_text(cleaned_text, encoding="utf-8")
            cleaned += 1

    # Xóa thư mục rỗng còn sót
    for d in sorted(BASE.rglob("*"), key=lambda p: -len(p.parts)):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    print(f"Removed archive pages  : {removed_archive}")
    print(f"Removed short pages    : {removed_short}")
    print(f"Cleaned content files  : {cleaned}")
    print(f"Remaining pages        : {len(list(BASE.rglob('metadata.json')))}")


if __name__ == "__main__":
    main()
