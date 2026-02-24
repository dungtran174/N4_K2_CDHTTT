"""
PTIT Multi-Domain Crawler
Crawl nội dung từ:
  - daotao.ptit.edu.vn  (chương trình đào tạo đại học)
  - tuyensinh.ptit.edu.vn (đề án tuyển sinh + học bổng)

Hỗ trợ: text, bảng (tablemarkdown), ảnh scan/OCR (pytesseract hoặc easyocr)

Usage:
    python crawl_page.py               # crawl theo CRAWL_TARGETS
    python crawl_page.py --url <URL>   # crawl một trang cụ thể
    python crawl_page.py --no-playwright
"""

import argparse
import hashlib
import io
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urljoin, urlparse

import requests
import unicodedata
from bs4 import BeautifulSoup, Tag
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Optional Playwright
try:
    from playwright.sync_api import sync_playwright
    from playwright.sync_api import TimeoutError as PWTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# OCR backends
OCR_BACKEND = None
try:
    import pytesseract
    from PIL import Image as PILImage
    pytesseract.get_tesseract_version()
    OCR_BACKEND = "tesseract"
except Exception:
    pass

if OCR_BACKEND is None:
    try:
        import easyocr
        from PIL import Image as PILImage
        _easyocr_reader = None
        OCR_BACKEND = "easyocr"
    except ImportError:
        pass

if OCR_BACKEND is None:
    try:
        from PIL import Image as PILImage
        OCR_BACKEND = "pillow_only"
    except ImportError:
        pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("crawler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ptit_crawler")
logger.info(f"OCR backend: {OCR_BACKEND or 'none'}")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "crawled_data"

CONFIG = {
    "output_dir": str(OUTPUT_DIR),
    "request_timeout": 30,
    "rate_limit_delay": 0.8,
    "playwright_timeout": 25000,
    "use_playwright": True,
    "min_image_bytes": 5000,
    "headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    },
}

# (seed_url, max_pages, allowed_path_prefixes or None)
CRAWL_TARGETS: List[Tuple[str, int, Optional[List[str]]]] = [
    (
        "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/",
        300,
        ["/chuong-trinh-dao-tao"],
    ),
    (
        "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh/thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2025/",
        150,
        ["/de-an-tuyen-sinh"],
    ),
    (
        "https://tuyensinh.ptit.edu.vn/",
        200,
        [
            "/hoc-bong",
            "/de-an-tuyen-sinh",
            "/thong-bao",
            "/thong-tin-tuyen-sinh",
        ],
    ),
]

_SKIP_PATTERNS = [
    re.compile(r"/\d{4}/\d{2}(/\d{2})?/?$"),
    re.compile(r"/(wp-admin|wp-json|wp-login|feed)/"),
    re.compile(r"/(author|tag|page)/"),
    re.compile(r"\?(page|paged|replytocom)="),
    re.compile(r"\.(jpg|jpeg|png|gif|svg|ico|css|js|woff2?|pdf)$", re.IGNORECASE),
]


def should_skip(url: str) -> bool:
    return any(p.search(url) for p in _SKIP_PATTERNS)


def url_allowed(url: str, allowed_prefixes: Optional[List[str]]) -> bool:
    if not allowed_prefixes:
        return True
    path = urlparse(url).path
    return any(path.startswith(pfx) for pfx in allowed_prefixes)


def sanitize(seg: str) -> str:
    seg = unquote(seg).strip()
    seg = unicodedata.normalize("NFKD", seg).encode("ascii", "ignore").decode("ascii")
    seg = re.sub(r'[\s/\\:*?"<>|%]+', "-", seg)
    seg = re.sub(r"-+", "-", seg).strip("-_")
    if len(seg) > 60:
        seg = seg[:51] + "-" + hashlib.md5(seg.encode()).hexdigest()[:8]
    return seg or "root"


def url_to_dir(url: str) -> Path:
    p = urlparse(url)
    path = Path(CONFIG["output_dir"]) / p.netloc
    for part in [s for s in p.path.split("/") if s]:
        path = path / sanitize(part)
    return path


_pw_ctx = None


def _get_pw_context():
    global _pw_ctx
    if _pw_ctx is None:
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=CONFIG["headers"]["User-Agent"],
            ignore_https_errors=True,
        )
        _pw_ctx = (pw, browser, ctx)
    return _pw_ctx


def _reset_pw_context():
    global _pw_ctx
    if _pw_ctx:
        try:
            _pw_ctx[2].close(); _pw_ctx[1].close(); _pw_ctx[0].stop()
        except Exception:
            pass
        _pw_ctx = None


def fetch_playwright(url: str) -> Optional[BeautifulSoup]:
    if not PLAYWRIGHT_AVAILABLE:
        return None
    for attempt in range(2):
        try:
            _, _, ctx = _get_pw_context()
            page = ctx.new_page()
            try:
                wait = "networkidle" if attempt == 0 else "domcontentloaded"
                page.goto(url, wait_until=wait, timeout=CONFIG["playwright_timeout"])
                for _ in range(3):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(600)
                html = page.content()
            finally:
                page.close()
            return BeautifulSoup(html, "html.parser")
        except PWTimeout:
            logger.warning(f"[PW] Timeout attempt {attempt+1}: {url}")
            if attempt == 0:
                continue
        except Exception as e:
            if "closed" in str(e).lower() or "crashed" in str(e).lower():
                _reset_pw_context()
                if attempt == 0:
                    continue
            logger.warning(f"[PW] Error {url}: {e}")
    return None


def fetch_requests(url: str) -> Optional[BeautifulSoup]:
    try:
        resp = requests.get(url, headers=CONFIG["headers"],
                            verify=False, timeout=CONFIG["request_timeout"])
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return None
        resp.encoding = resp.apparent_encoding or "utf-8"
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logger.error(f"[REQ] Error {url}: {e}")
        return None


def fetch(url: str) -> Optional[BeautifulSoup]:
    if CONFIG.get("use_playwright") and PLAYWRIGHT_AVAILABLE:
        soup = fetch_playwright(url)
        if soup:
            return soup
    return fetch_requests(url)


def _init_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("[OCR] Initialising EasyOCR (vi+en)...")
        _easyocr_reader = easyocr.Reader(["vi", "en"], gpu=False)
    return _easyocr_reader


def ocr_image_bytes(data: bytes, source_url: str = "") -> str:
    if not OCR_BACKEND or OCR_BACKEND == "pillow_only":
        return ""
    if len(data) < CONFIG["min_image_bytes"]:
        return ""
    try:
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if w < 80 or h < 40:
            return ""
        if OCR_BACKEND == "tesseract":
            text = pytesseract.image_to_string(img, lang="vie+eng", config="--psm 6")
        elif OCR_BACKEND == "easyocr":
            reader = _init_easyocr_reader()
            results = reader.readtext(data, detail=0, paragraph=True, text_threshold=0.5)
            text = "\n".join(results)
        else:
            return ""
        text = text.strip()
        if len(text) < 10:
            return ""
        logger.info(f"  [OCR] {len(text)} chars from {source_url}")
        return text
    except Exception as e:
        logger.debug(f"[OCR] Failed on {source_url}: {e}")
        return ""


def fetch_image_bytes(url: str) -> bytes:
    try:
        r = requests.get(url, headers=CONFIG["headers"], verify=False, timeout=15)
        r.raise_for_status()
        return r.content
    except Exception:
        return b""


_CONTENT_SELECTORS = [
    ".entry-content", ".post-content", ".page-content", ".post-body",
    "article", "main", '[role="main"]',
    "#content", "#main", "#primary", ".content-area",
    ".single-content", ".page",
]


def extract_images_ocr(soup: BeautifulSoup, base_url: str) -> List[str]:
    if OCR_BACKEND in (None, "pillow_only"):
        return []
    texts: List[str] = []
    seen_urls: Set[str] = set()
    container = None
    for sel in _CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 50:
            container = el
            break
    container = container or soup.find("body") or soup
    for img in container.find_all("img"):
        src = img.get("data-src") or img.get("src") or ""
        if not src or src.startswith("data:"):
            continue
        abs_url = urljoin(base_url, src)
        if abs_url in seen_urls:
            continue
        seen_urls.add(abs_url)
        img_bytes = fetch_image_bytes(abs_url)
        if img_bytes:
            text = ocr_image_bytes(img_bytes, abs_url)
            if text:
                texts.append(f"<!-- OCR: {abs_url} -->\n{text}")
    return texts


_REMOVE_SELECTORS = [
    "script", "style", "noscript",
    "footer", ".footer", "#footer", '[class*="footer"]',
    ".wrap_footer", ".ovamegamenu_container_default",
    "#scrollUp", ".back-to-top",
    ".elementor-screen-only",
    "aside.sidebar", ".sidebar", ".widget", ".widget-area",
    "nav", ".nav", ".navigation", ".breadcrumb",
    ".social-share", ".share-buttons",
    '[class*="elementor-widget-gimont_elementor_menu"]',
    '[class*="elementor-widget-ova_logo"]',
    '[class*="elementor-widget-social-icons"]',
    '[class*="contact-info"]',
    ".site-footer", ".footer-widgets", ".footer-bottom",
    ".copyright", "#colophon",
]

_NOISE_RE = [
    re.compile(r"^-\s*$", re.MULTILINE),
    re.compile(r"^-\s*(Chi tiet bai viet|Chia se:|Share|Tags?:)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"\b(admintuyen\w*|admindaotao|admintuyensinh|ptitadmin)\b", re.IGNORECASE),
]

_FOOTER_WORDS = re.compile(
    r"(\u00a9\s?\d{4}|Học viện Công nghệ Bưu chính Viễn thông"
    r"|Số \d+.*(Trần Phú|Hà Đông|Mai Dịch)"
    r"|Tuyển sinh Học viện.*hotline"
    r"|Liên hệ tuyển sinh)",
    re.IGNORECASE,
)


def _table_to_md(table: Tag) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""
    lines: List[str] = []
    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        texts = [" ".join(c.get_text(separator=" ", strip=True).split()) for c in cells]
        lines.append("| " + " | ".join(texts) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(lines)


def extract_markdown(soup: BeautifulSoup, page_url: str = "") -> str:
    for sel in _REMOVE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()

    container = None
    for sel in _CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 100:
            container = el
            break
    container = container or soup.find("body") or soup

    parts: List[str] = []
    seen: Set[str] = set()
    done_tables: Set[int] = set()

    for el in container.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table"],
        recursive=True,
    ):
        tag = el.name

        if tag == "table":
            eid = id(el)
            if eid in done_tables or el.find_parent("table"):
                continue
            md = _table_to_md(el)
            if md and md not in seen:
                seen.add(md)
                done_tables.add(eid)
                for t in el.find_all("table"):
                    done_tables.add(id(t))
                parts.append(md)
            continue

        if el.find_parent("table") and id(el.find_parent("table")) in done_tables:
            continue

        # For <p> tags with <br> separators, split on <br> into individual lines
        if tag == "p" and el.find("br"):
            import copy as _copy
            try:
                el_copy = _copy.copy(el)
                for br in el_copy.find_all("br"):
                    br.replace_with("\n")
                raw_lines = el_copy.get_text().split("\n")
                br_lines = [" ".join(l.split()) for l in raw_lines]
                br_lines = [l for l in br_lines if len(l) >= 5
                            and not (_FOOTER_WORDS.search(l) and len(l) < 400)]
                if br_lines:
                    combined = "\n".join(br_lines)
                    key = combined[:200]
                    if key not in seen:
                        seen.add(key)
                        parts.append(combined)
            except Exception:
                pass
            continue

        text = " ".join(el.get_text(separator=" ", strip=True).split())
        if not text or len(text) < 5:
            continue
        # Only treat as footer/noise if text is SHORT — long text is article content
        if _FOOTER_WORDS.search(text) and len(text) < 400:
            continue
        key = text[:200]
        if key in seen:
            continue
        seen.add(key)

        if tag.startswith("h"):
            parts.append(f"{'#' * int(tag[1])} {text}")
        elif tag == "p":
            parts.append(text)
        elif tag in ("ul", "ol"):
            items = []
            for i, li in enumerate(el.find_all("li", recursive=False), 1):
                t = " ".join(li.get_text(separator=" ", strip=True).split())
                if t:
                    prefix = f"{i}." if tag == "ol" else "-"
                    items.append(f"{prefix} {t}")
            if items:
                parts.append("\n".join(items))

    if page_url and OCR_BACKEND not in (None, "pillow_only"):
        ocr_blocks = extract_images_ocr(soup, page_url)
        if ocr_blocks:
            parts.append("\n\n---\n## Nội dung ảnh (OCR)\n")
            parts.extend(ocr_blocks)

    content = "\n\n".join(parts)
    for pat in _NOISE_RE:
        content = pat.sub("", content)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


def get_links(
    soup: BeautifulSoup,
    base_url: str,
    allowed_netloc: str,
    allowed_prefixes: Optional[List[str]],
) -> Set[str]:
    links: Set[str] = set()
    for a in soup.find_all("a", href=True):
        raw = a["href"].strip()
        if not raw or raw.startswith(("javascript:", "mailto:", "tel:")):
            continue
        if raw.startswith("www."):
            raw = "https://" + raw
        try:
            href = urljoin(base_url, raw).split("#")[0].rstrip("?")
        except Exception:
            continue
        p = urlparse(href)
        if p.scheme not in ("http", "https"):
            continue
        if allowed_netloc not in p.netloc:
            continue
        if should_skip(href):
            continue
        if not url_allowed(href, allowed_prefixes):
            continue
        links.add(href)
    return links


def save_page(url: str, title: str, content: str) -> None:
    out_dir = url_to_dir(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "content.md").write_text(content, encoding="utf-8")
    meta = {
        "url": url,
        "title": title,
        "crawl_date": datetime.now().isoformat(),
        "content_file": "content.md",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"  Saved -> {out_dir.relative_to(REPO_ROOT)}")


def crawl_single(url: str) -> bool:
    soup = fetch(url)
    if not soup:
        return False
    title = (soup.title.string or "Untitled").strip() if soup.title else "Untitled"
    content = extract_markdown(soup, page_url=url)
    if not content:
        logger.warning(f"No content: {url}")
        return False
    save_page(url, title, content)
    return True


def crawl_domain(
    start_url: str,
    max_pages: int = 300,
    allowed_prefixes: Optional[List[str]] = None,
):
    visited: Set[str] = set()
    queue: List[str] = [start_url]
    netloc = urlparse(start_url).netloc
    ok = skipped = failed = 0

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if should_skip(url):
            skipped += 1
            continue

        soup = fetch(url)
        if not soup:
            failed += 1
            continue

        title = (soup.title.string or "Untitled").strip() if soup.title else "Untitled"

        if url_allowed(url, allowed_prefixes):
            content = extract_markdown(soup, page_url=url)
            if content and len(content) > 50:
                save_page(url, title, content)
                ok += 1
            else:
                logger.info(f"[EMPTY] {url}")

        for link in get_links(soup, url, netloc, allowed_prefixes):
            if link not in visited:
                queue.append(link)

        time.sleep(CONFIG["rate_limit_delay"])

    logger.info(
        f"Done [{netloc}]: saved={ok} skipped={skipped} failed={failed} visited={len(visited)}"
    )


def main():
    parser = argparse.ArgumentParser(description="PTIT Multi-Domain Crawler")
    parser.add_argument("--url", help="Crawl một URL cụ thể")
    parser.add_argument("--no-playwright", action="store_true")
    parser.add_argument("--no-ocr", action="store_true", help="Tắt OCR ảnh")
    args = parser.parse_args()

    if args.no_playwright:
        CONFIG["use_playwright"] = False

    if args.no_ocr:
        global OCR_BACKEND
        OCR_BACKEND = None
        logger.info("OCR disabled by --no-ocr flag.")

    if args.url:
        ok = crawl_single(args.url)
        logger.info("SUCCESS" if ok else "FAILED")
        return

    for seed_url, max_p, prefixes in CRAWL_TARGETS:
        logger.info(
            f"\n{'='*60}\nCrawling: {seed_url}\nMax: {max_p}  Prefixes: {prefixes}\n{'='*60}"
        )
        crawl_domain(seed_url, max_pages=max_p, allowed_prefixes=prefixes)

    _reset_pw_context()
    logger.info("All done.")


if __name__ == "__main__":
    main()