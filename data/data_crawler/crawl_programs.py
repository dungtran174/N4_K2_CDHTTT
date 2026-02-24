# -*- coding: utf-8 -*-
"""
crawl_programs.py
Crawl đặc biệt cho các trang chương trình đào tạo của daotao.ptit.edu.vn
Trang dùng JS + section ID (#tong_quan, #chuan_dau_ra...) thay vì <p> tags thông thường.
"""
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import unquote, urlparse

import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("crawl_programs.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ptit_programs")

from playwright.sync_api import sync_playwright
from playwright.sync_api import TimeoutError as PWTimeout
from bs4 import BeautifulSoup, Tag

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "crawled_data"

# 27 chương trình đào tạo cần crawl
PROGRAM_URLS = [
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/tri-tue-nhan-tao-van-vat-aiot/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-cong-nghe-thong-tin-dinh-huong-ung-dung/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/phan-tich-du-lieu-trong-tai-chinh-kinh-doanh/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-tri-tue-nhan-tao/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-thiet-ke-va-phat-trien-game/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-ky-thuat-dien-tu-vien-thong/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-an-toan-thong-tin/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-ky-thuat-dien-dien-tu/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-da-phuong-tien/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-truyen-thong-da-phuong-tien/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-marketing/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-quan-tri-kinh-doanh/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-thong-tin/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-ke-toan/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-thuong-mai-dien-tu/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-tai-chinh-fintech/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-ky-thuat-dieu-khien-va-tu-dong-hoa/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-khoa-hoc-may-tinh/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-bao-chi-journalism/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-cong-nghe-thong-tin-he-clc/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-ky-thuat-du-lieu-nganh-mang-may-tinh-va-truyen-thong-du-lieu/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-marketing-he-clc/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/nganh-ke-toan-chat-luong-cao-chuan-quoc-te-acca/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-cong-nghe-thong-tin-viet-nhat/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-quan-he-cong-chung-nganh-marketing/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-logistics-va-quan-tri-chuoi-cung-ung/",
    "https://daotao.ptit.edu.vn/chuong-trinh-dao-tao/chuong-trinh-an-toan-thong-tin-chat-luong-cao/",
]

# Section IDs có trong mỗi trang chương trình
SECTION_IDS = [
    "tong_quan",
    "chuan_dau_ra",
    "cau_truc_chuong_trinh",
    "nghe_nghiep",
    "hoc_phi",
    "dieu_kien_tuyen_sinh",
    "quy_trinh_nhap_hoc",
    "tai_lieu_dao_tao",
]

SECTION_NAMES = {
    "tong_quan": "Tổng quan",
    "chuan_dau_ra": "Chuẩn đầu ra",
    "cau_truc_chuong_trinh": "Cấu trúc chương trình",
    "nghe_nghiep": "Nghề nghiệp",
    "hoc_phi": "Học phí",
    "dieu_kien_tuyen_sinh": "Điều kiện tuyển sinh",
    "quy_trinh_nhap_hoc": "Quy trình nhập học",
    "tai_lieu_dao_tao": "Tài liệu đào tạo",
}

# ── Path helpers ──────────────────────────────────────────────────────────────

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
    path = OUTPUT_DIR / p.netloc
    for part in [s for s in p.path.split("/") if s]:
        path = path / sanitize(part)
    return path


# ── Playwright fetch (click all tabs) ────────────────────────────────────────

_pw_ctx = None

def get_pw_context():
    global _pw_ctx
    if _pw_ctx is None:
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
            ),
            ignore_https_errors=True,
        )
        _pw_ctx = (pw, browser, ctx)
    return _pw_ctx


def fetch_program_page(url: str) -> Optional[BeautifulSoup]:
    """Fetch program page and click all tabs to expose hidden content."""
    for attempt in range(2):
        try:
            _, _, ctx = get_pw_context()
            page = ctx.new_page()
            try:
                wait = "networkidle" if attempt == 0 else "domcontentloaded"
                page.goto(url, wait_until=wait, timeout=25000)

                # Scroll to load lazy content
                for _ in range(3):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(500)

                # Click all tab buttons to reveal hidden tab content
                tabs = page.query_selector_all(".nav-tab, [role='tab'], .tab-btn, .tab-link")
                logger.info(f"  Found {len(tabs)} tab buttons")
                for tab in tabs:
                    try:
                        tab.click()
                        page.wait_for_timeout(300)
                    except Exception:
                        pass

                # Final scroll after clicking
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(500)

                html = page.content()
            finally:
                page.close()
            return BeautifulSoup(html, "html.parser")
        except PWTimeout:
            logger.warning(f"[PW] Timeout attempt {attempt+1}: {url}")
            if attempt == 0:
                # Reset context and retry
                global _pw_ctx
                if _pw_ctx:
                    try:
                        _pw_ctx[2].close(); _pw_ctx[1].close(); _pw_ctx[0].stop()
                    except Exception:
                        pass
                    _pw_ctx = None
                continue
        except Exception as e:
            logger.error(f"[PW] Error {url}: {e}")
    return None


# ── Table → Markdown ──────────────────────────────────────────────────────────

def table_to_md(table: Tag) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""
    lines = []
    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        texts = [" ".join(c.get_text(separator=" ", strip=True).split()) for c in cells]
        lines.append("| " + " | ".join(texts) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(lines)


# ── Smart section extractor ───────────────────────────────────────────────────

def extract_section_text(section: Tag) -> str:
    """
    Extract all meaningful text from a section element.
    Handles: headings, paragraphs, lists, tables, and direct div text.
    """
    # Remove navigation elements inside the section (tab buttons etc.)
    for junk in section.select(".nav-tab, .breadcrumb, .tab-btn, .tab-link"):
        junk.decompose()

    parts: List[str] = []
    seen: Set[str] = set()
    done_tables: Set[int] = set()

    def _add(text: str):
        key = text[:150]
        if key not in seen and len(text) > 2:
            seen.add(key)
            parts.append(text)

    def walk(el: Tag, depth: int = 0):
        if el.name in ("script", "style", "noscript"):
            return

        tag = el.name

        if tag == "table":
            eid = id(el)
            if eid not in done_tables and not el.find_parent("table"):
                md = table_to_md(el)
                if md:
                    done_tables.add(eid)
                    for t in el.find_all("table"):
                        done_tables.add(id(t))
                    _add(md)
            return  # don't recurse into table children

        if el.find_parent("table") and id(el.find_parent("table")) in done_tables:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = " ".join(el.get_text(separator=" ", strip=True).split())
            if text:
                level = int(tag[1])
                _add(f"{'#' * level} {text}")
            return

        if tag == "p":
            text = " ".join(el.get_text(separator=" ", strip=True).split())
            if len(text) > 5:
                _add(text)
            return

        if tag in ("ul", "ol"):
            items = []
            for i, li in enumerate(el.find_all("li", recursive=False), 1):
                t = " ".join(li.get_text(separator=" ", strip=True).split())
                if t:
                    prefix = f"{i}." if tag == "ol" else "-"
                    items.append(f"{prefix} {t}")
            if items:
                _add("\n".join(items))
            return

        if tag == "br":
            return

        # For div/span/section: get direct text nodes, then recurse into children
        if tag in ("div", "span", "section", "article", "header", None):
            # Check for direct text content (not in child elements)
            direct_text_parts = []
            for child in el.children:
                if hasattr(child, "string") and child.string:
                    t = child.string.strip()
                    if len(t) > 3:
                        direct_text_parts.append(t)

            # If this div has substantial direct text AND no block children, treat as paragraph
            block_children = el.find_all(
                ["h1","h2","h3","h4","h5","h6","p","ul","ol","table","div"], recursive=False
            )
            if direct_text_parts and not block_children:
                text = " ".join(" ".join(direct_text_parts).split())
                if len(text) > 10:
                    _add(text)
            else:
                # Recurse into children
                for child in el.children:
                    if hasattr(child, "name") and child.name:
                        walk(child, depth + 1)

    walk(section)
    return "\n\n".join(parts)


# ── Main extractor for program pages ─────────────────────────────────────────

# Meta info selectors for the top info bar (mã ngành, thời gian, kỳ nhập học...)
_META_SELECTORS = [
    ".row_site", ".program-meta", ".dir-meta", "[class*='meta']",
    "[class*='info-bar']", "[class*='program-info']",
]

def extract_program_markdown(soup: BeautifulSoup, url: str) -> str:
    """
    Dedicated extractor for daotao.ptit.edu.vn program pages.
    Extracts: page title, meta info, and all section content by ID.
    """
    parts: List[str] = []

    # ── Page title ────────────────────────────────────────────────────────────
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = " ".join(h1.get_text(separator=" ", strip=True).split())
        if title:
            parts.append(f"# {title}")

    # ── Meta info bar (mã ngành, thời gian, cơ sở…) ─────────────────────────
    for sel in _META_SELECTORS:
        meta_el = soup.select_one(sel)
        if meta_el:
            meta_text = " ".join(meta_el.get_text(separator=" | ", strip=True).split())
            # Only include if it looks like meta data
            if any(kw in meta_text for kw in ["ngành", "năm", "Hà Nội", "TP", "học kỳ", "Mùa"]):
                key_pairs = re.sub(r"\s*\|\s*", " | ", meta_text)
                parts.append(f"**Thông tin:** {key_pairs}")
                break

    # ── Extract each section by ID ────────────────────────────────────────────
    found_any = False
    for sec_id in SECTION_IDS:
        section = soup.find(id=sec_id)
        if not section:
            continue
        found_any = True
        sec_name = SECTION_NAMES.get(sec_id, sec_id.replace("_", " ").title())
        section_text = extract_section_text(section)
        if section_text:
            parts.append(f"\n## {sec_name}\n\n{section_text}")

    # ── Fallback: use ova_dir_content if no sections found ───────────────────
    if not found_any:
        logger.warning(f"No section IDs found on {url}, falling back to .ova_dir_content")
        container = soup.select_one(".ova_dir_content, #main-content, .main")
        if container:
            # Remove footer/nav
            for el in container.select(
                "footer, .wrap_footer, nav, .navigation, .breadcrumb, script, style"
            ):
                el.decompose()
            text = container.get_text(separator="\n", strip=True)
            # Clean up excessive newlines
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > 50:
                parts.append(text)

    return "\n\n".join(parts)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_page(url: str, title: str, content: str) -> None:
    out_dir = url_to_dir(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "content.md").write_text(content, encoding="utf-8")
    meta = {
        "url": url,
        "title": title,
        "crawl_date": datetime.now().isoformat(),
        "content_file": "content.md",
        "source": "daotao.ptit.edu.vn",
        "type": "chuong_trinh_dao_tao",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"  Saved -> {out_dir.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ok = failed = 0
    for url in PROGRAM_URLS:
        url = url.replace("http://", "https://")  # normalize
        logger.info(f"\nCrawling: {url}")
        soup = fetch_program_page(url)
        if not soup:
            logger.error(f"  FAILED to fetch: {url}")
            failed += 1
            continue

        title = ""
        h1 = soup.find("h1")
        if h1:
            title = " ".join(h1.get_text(separator=" ", strip=True).split())
        if not title and soup.title:
            title = (soup.title.string or "").split("|")[0].strip()

        content = extract_program_markdown(soup, url)
        logger.info(f"  Content length: {len(content)} chars")

        if len(content) < 100:
            logger.warning(f"  VERY SHORT content ({len(content)} chars) for {url}")

        save_page(url, title or "Chương trình đào tạo", content)
        ok += 1
        time.sleep(0.5)

    # Close Playwright
    global _pw_ctx
    if _pw_ctx:
        try:
            _pw_ctx[2].close(); _pw_ctx[1].close(); _pw_ctx[0].stop()
        except Exception:
            pass

    logger.info(f"\nDone: ok={ok} failed={failed}")


if __name__ == "__main__":
    main()
