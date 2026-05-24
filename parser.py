"""PDF structure parser: extracts readable text blocks, classifying each as
title / heading / body while skipping references, captions, page numbers,
table of contents, author lists, and figure artefacts.

Also provides extract_references() and extract_figure_images() helpers.
"""

import base64
import re
from collections import Counter
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Block:
    type: str       # 'title' | 'heading' | 'body'
    text: str       # display text — hyphen-fixed, keeps [N] citations
    tts_text: str   # TTS text — no [N], cleaned for speech
    page: int


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _fix_hyphens(text: str) -> str:
    """Join hyphenated line-breaks: 'transac- tion' → 'transaction'."""
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", text)


def _strip_citations(text: str) -> str:
    """Remove citation bracket sequences: [1], [2,3], [ 61 ], [1–3]."""
    return re.sub(r"\s*\[\s*[\d,;\s–—\-]+\s*\]", "", text)


def clean_for_tts(text: str) -> str:
    """Full cleaning pipeline for TTS (hyphen-fix + strip citations)."""
    text = _fix_hyphens(text)
    text = _strip_citations(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_TOC_ENTRY = re.compile(r"^([IVX]+\.|[A-Z]\.|[0-9]+\.)\s+.{2,}\d+\s*$")
_SECTION_LABEL = re.compile(r"^([IVX]+|[A-Z])\.\s+[A-Z][A-Z\s\-]{2,}$")
_FIGURE_CAP = re.compile(r"^(fig(ure)?|table|algorithm|listing)\s*\.?\s*\d+", re.I)
_AUTHOR_YEAR = re.compile(r"\w+'[0-9]{2}")
_CHART_AXIS = re.compile(r"^\d+[MKGBT]$")
_DATE_SERIES = re.compile(r"^(\d{4}\s+){2,}")
_TIME_SERIES = re.compile(r"^[\d\s]+Time\s+Since")


def _is_toc_entry(text: str) -> bool:
    return bool(_TOC_ENTRY.match(text))


def _is_figure_noise(text: str, bbox: tuple, col_width: float) -> bool:
    w = bbox[2] - bbox[0]
    if w < 100:
        return True
    if _AUTHOR_YEAR.search(text):
        return True
    if _DATE_SERIES.match(text) or _TIME_SERIES.match(text):
        return True
    words = text.split()
    axis_tokens = sum(1 for wd in words if _CHART_AXIS.match(wd))
    if words and axis_tokens / len(words) > 0.35:
        return True
    if col_width > 0 and w < 0.87 * col_width:
        return True
    return False


# ---------------------------------------------------------------------------
# Per-page spatial preprocessing
# ---------------------------------------------------------------------------

def _build_page_filter(raw_page_blocks: list[dict], col_width: float) -> set[int]:
    skip_idx: set[int] = set()
    caption_ys: list[float] = []
    for b in raw_page_blocks:
        if _FIGURE_CAP.match(b["text"]):
            caption_ys.append(b["bbox"][1])
    if not caption_ys:
        return skip_idx
    for cap_y in caption_ys:
        for i, b in enumerate(raw_page_blocks):
            if b["bbox"][3] <= cap_y + 2:
                if _is_figure_noise(b["text"], b["bbox"], col_width):
                    skip_idx.add(i)
    return skip_idx


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_pdf(path: str, include_captions: bool = False,
              max_blocks: int | None = None) -> list[Block]:
    doc = fitz.open(path)
    raw = []

    for page_num, page in enumerate(doc):
        d = page.get_text("dict", flags=0)
        page_raw: list[dict] = []
        for blk in d["blocks"]:
            if blk["type"] != 0:
                continue
            spans = [s for line in blk["lines"] for s in line["spans"]]
            if not spans:
                continue
            text = " ".join(s["text"].strip() for s in spans if s["text"].strip())
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            avg_size = sum(s["size"] for s in spans) / len(spans)
            is_bold = any(s["flags"] & 16 for s in spans)
            page_raw.append({
                "text": text, "size": avg_size, "bold": is_bold,
                "page": page_num, "bbox": blk["bbox"],
            })
        raw.extend(page_raw)

    if not raw:
        return []

    sizes = [round(b["size"]) for b in raw if len(b["text"]) > 20]
    body_size = Counter(sizes).most_common(1)[0][0] if sizes else 12

    widths = sorted(b["bbox"][2] - b["bbox"][0] for b in raw if len(b["text"]) > 40)
    col_width = widths[int(len(widths) * 0.9)] if widths else 500

    page_blocks: dict[int, list] = {}
    for b in raw:
        page_blocks.setdefault(b["page"], []).append(b)

    toc_pages: set[int] = set()
    for pg, pblocks in page_blocks.items():
        texts = [b["text"] for b in pblocks if len(b["text"]) > 4]
        has_toc_header = any(t.upper() in ("CONTENTS", "TABLE OF CONTENTS") for t in texts)
        meaningful = [t for t in texts if len(t) > 8]
        toc_count = sum(1 for t in meaningful if _is_toc_entry(t))
        ratio = toc_count / len(meaningful) if meaningful else 0
        if has_toc_header or ratio > 0.4:
            toc_pages.add(pg)

    noise_texts: set[int] = set()
    for pg, pblocks in page_blocks.items():
        for i in _build_page_filter(pblocks, col_width):
            noise_texts.add(id(pblocks[i]))

    blocks: list[Block] = []
    in_references = False
    title_emitted = False

    for b in raw:
        text, size, is_bold, page = b["text"], b["size"], b["bold"], b["page"]

        if page in toc_pages:
            continue
        if id(b) in noise_texts:
            continue

        if re.match(
            r"^(references?|bibliography|works cited|appendix)(\s+\d+)?$", text, re.I
        ):
            in_references = True
            continue
        if in_references:
            continue

        if re.match(r"^(page\s+)?\d+(\s+of\s+\d+)?$", text, re.I):
            continue
        if re.match(r"^(https?://|doi:|arxiv:|www\.|[a-z0-9._%+\-]+@\S+)", text, re.I):
            continue
        if re.match(
            r"^\d+\s+(department|university|institute|lab|school|college|center|"
            r"google|apple|meta|microsoft|amazon|openai|deepmind|berkeley|mit|stanford)",
            text, re.I,
        ):
            continue
        if re.match(r"^[∗†‡§¶✉]\s*\S", text):
            continue
        if len(text) < 6:
            continue

        is_caption = bool(_FIGURE_CAP.match(text))
        if is_caption and not include_captions:
            continue

        if page < 3:
            comma_ratio = text.count(",") / max(len(text.split()), 1)
            digit_count = sum(c.isdigit() for c in text)
            if comma_ratio > 0.15 and digit_count > 2 and not text.endswith("."):
                continue

        # Fix hyphens first; keep [N] in display text, strip for TTS
        display_text = re.sub(r"\s+", " ", _fix_hyphens(text)).strip()
        tts_text = re.sub(r"\s+", " ", _strip_citations(display_text)).strip()
        if not tts_text:
            continue

        is_section_label = bool(_SECTION_LABEL.match(display_text))
        is_heading = (
            is_section_label
            or size > body_size + 1.5
            or (is_bold and size >= body_size - 0.5 and len(display_text) < 120)
        )

        if not title_emitted and (size > body_size + 1 or is_heading):
            blocks.append(Block("title", display_text, tts_text, page))
            title_emitted = True
        elif is_heading:
            blocks.append(Block("heading", display_text, tts_text, page))
        else:
            blocks.append(Block("body", display_text, tts_text, page))
            title_emitted = True

        if max_blocks is not None and len(blocks) >= max_blocks:
            break

    return blocks


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

def extract_references(path: str) -> dict[int, str]:
    """Parse the References section and return {number: full_text}.

    Handles two layouts:
    - Explicit header: a block matching /^references?$/i before the entries
    - Implicit: first occurrence of a '[1] ...' block (no header block)
    Also handles multiple refs packed into one wide block.
    """
    doc = fitz.open(path)
    # Collect all text blocks with page numbers
    all_blocks: list[tuple[int, str]] = []
    for pg_num, page in enumerate(doc):
        d = page.get_text("dict", flags=0)
        for blk in d["blocks"]:
            if blk["type"] != 0:
                continue
            spans = [s for line in blk["lines"] for s in line["spans"]]
            raw = " ".join(s["text"].strip() for s in spans if s["text"].strip())
            text = re.sub(r"\s+", " ", raw).strip()
            if text:
                all_blocks.append((pg_num, text))

    # Find where references start
    ref_start = None
    for i, (pg, text) in enumerate(all_blocks):
        if re.match(r"^references?$", text, re.I):
            ref_start = i + 1
            break
    if ref_start is None:
        # Fall back: first block that starts with [1] on a later page (>= page 30)
        for i, (pg, text) in enumerate(all_blocks):
            if pg >= 30 and re.match(r"^\[1\]\s", text):
                ref_start = i
                break

    if ref_start is None:
        return {}

    # Parse reference entries — multiple refs may be packed in one block
    refs: dict[int, str] = {}
    current_num: int | None = None
    current_parts: list[str] = []

    def _flush():
        if current_num is not None and current_parts:
            refs[current_num] = " ".join(current_parts).strip()

    # Split each block into individual [N] entries
    _REF_SPLIT = re.compile(r"(?=\[\d+\]\s)")

    for pg, text in all_blocks[ref_start:]:
        # Skip page numbers and short noise
        if re.match(r"^\d+$", text):
            continue
        # Split block on [N] boundaries (handles packed refs in one block)
        parts = _REF_SPLIT.split(text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^\[(\d+)\]\s+(.*)", part, re.S)
            if m:
                _flush()
                current_num = int(m.group(1))
                current_parts = [m.group(2).strip()]
            elif current_num is not None:
                current_parts.append(part)

    _flush()
    return refs


# ---------------------------------------------------------------------------
# Figure image extraction
# ---------------------------------------------------------------------------

def extract_figure_images(path: str) -> dict[int, str]:
    """Render figure regions as base64 PNG, keyed by figure number.

    For each 'Figure N.' caption block, renders the page area above the
    caption at 2× resolution.  Returns {figure_number: base64_png_string}.
    """
    doc = fitz.open(path)
    figures: dict[int, str] = {}

    for page in doc:
        d = page.get_text("dict", flags=0)
        page_raw = d["blocks"]

        # Collect full-width body text bboxes to find where figures start
        # (we want to exclude body text that sits above the figure on the same page)
        full_width_ys: list[float] = []
        page_w = page.rect.width

        for blk in page_raw:
            if blk["type"] != 0:
                continue
            bw = blk["bbox"][2] - blk["bbox"][0]
            spans = [s for line in blk["lines"] for s in line["spans"]]
            raw = " ".join(s["text"].strip() for s in spans if s["text"].strip())
            if len(raw) > 40 and bw > page_w * 0.6:
                full_width_ys.append(blk["bbox"][3])

        for blk in page_raw:
            if blk["type"] != 0:
                continue
            spans = [s for line in blk["lines"] for s in line["spans"]]
            raw = " ".join(s["text"].strip() for s in spans if s["text"].strip())
            text = re.sub(r"\s+", " ", raw).strip()

            m = re.match(r"^Fig(?:ure)?\s*\.?\s*(\d+)", text, re.I)
            if not m:
                continue

            fig_num = int(m.group(1))
            cap_top = blk["bbox"][1]

            # Figure starts just after the last full-width body text above caption
            body_above = [y for y in full_width_ys if y < cap_top - 10]
            fig_top = max(body_above) + 4 if body_above else 30.0

            if cap_top - fig_top < 20:
                continue

            clip = fitz.Rect(20, fig_top, page_w - 20, cap_top - 4)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
            figures[fig_num] = base64.b64encode(pix.tobytes("png")).decode()

    return figures
