"""PDF structure parser: extracts readable text blocks, classifying each as
title / heading / body while skipping references, captions, page numbers,
table of contents, author lists, and figure artefacts."""

import re
from collections import Counter
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Block:
    type: str   # 'title' | 'heading' | 'body'
    text: str
    page: int


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_TOC_ENTRY = re.compile(
    r"^([IVX]+\.|[A-Z]\.|[0-9]+\.)\s+.{2,}\d+\s*$"
)
_SECTION_LABEL = re.compile(
    r"^([IVX]+|[A-Z])\.\s+[A-Z][A-Z\s\-]{2,}$"   # "II. INTRODUCTION"
)
_FIGURE_CAP = re.compile(
    r"^(fig(ure)?|table|algorithm|listing)\s*\.?\s*\d+", re.I
)
_AUTHOR_YEAR  = re.compile(r"\w+'[0-9]{2}")           # Zalka'04, Häner+'20
_CHART_AXIS   = re.compile(r"^\d+[MKGBT]$")           # 10M, 100B, 1T ...
_DATE_SERIES  = re.compile(r"^(\d{4}\s+){2,}")        # 2010 2012 2014 ...
_TIME_SERIES  = re.compile(r"^[\d\s]+Time\s+Since")   # "0 5 10 ... Time Since"


def _is_toc_entry(text: str) -> bool:
    return bool(_TOC_ENTRY.match(text))


def _is_figure_noise(text: str, bbox: tuple, col_width: float) -> bool:
    """True for inline figure text (axis labels, legends, chart titles)."""
    w = bbox[2] - bbox[0]
    # Very narrow block → axis / annotation
    if w < 100:
        return True
    # Author-year citation style found in chart legends
    if _AUTHOR_YEAR.search(text):
        return True
    # Date / time series (chart x-axis)
    if _DATE_SERIES.match(text) or _TIME_SERIES.match(text):
        return True
    # Majority of words are SI-suffixed numbers (axis ticks)
    words = text.split()
    axis_tokens = sum(1 for wd in words if _CHART_AXIS.match(wd))
    if words and axis_tokens / len(words) > 0.35:
        return True
    # Block is significantly narrower than text column
    if col_width > 0 and w < 0.87 * col_width:
        return True
    return False


# ---------------------------------------------------------------------------
# Per-page spatial preprocessing
# ---------------------------------------------------------------------------

def _build_page_filter(raw_page_blocks: list[dict], col_width: float) -> set[int]:
    """Return indices (into raw_page_blocks) that are inside figure regions.

    A figure region is everything above a figure/table caption that is not
    full-column-width body text.
    """
    skip_idx: set[int] = set()

    # Find caption y-starts
    caption_ys: list[float] = []
    for b in raw_page_blocks:
        if _FIGURE_CAP.match(b["text"]):
            caption_ys.append(b["bbox"][1])   # top y of caption

    if not caption_ys:
        return skip_idx

    for cap_y in caption_ys:
        for i, b in enumerate(raw_page_blocks):
            # Block ends before the caption starts → candidate figure content
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

    # --- Global stats ---
    sizes = [round(b["size"]) for b in raw if len(b["text"]) > 20]
    body_size = Counter(sizes).most_common(1)[0][0] if sizes else 12

    # Estimate text column width from the widest body-text blocks
    widths = sorted(
        b["bbox"][2] - b["bbox"][0]
        for b in raw
        if len(b["text"]) > 40
    )
    col_width = widths[int(len(widths) * 0.9)] if widths else 500

    # --- ToC page detection ---
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

    # --- Per-page figure-noise filter ---
    noise_texts: set[int] = set()  # use id(b) as key
    for pg, pblocks in page_blocks.items():
        local_skip = _build_page_filter(pblocks, col_width)
        for i in local_skip:
            noise_texts.add(id(pblocks[i]))

    # --- Main classification pass ---
    blocks: list[Block] = []
    in_references = False
    title_emitted = False

    for b in raw:
        text, size, is_bold, page = b["text"], b["size"], b["bold"], b["page"]

        # Skip ToC pages
        if page in toc_pages:
            continue

        # Skip inline figure noise
        if id(b) in noise_texts:
            continue

        # Stop at references / appendix
        if re.match(
            r"^(references?|bibliography|works cited|appendix)(\s+\d+)?$",
            text, re.I,
        ):
            in_references = True
            continue
        if in_references:
            continue

        # Skip page numbers
        if re.match(r"^(page\s+)?\d+(\s+of\s+\d+)?$", text, re.I):
            continue

        # Skip URLs, DOIs, emails
        if re.match(r"^(https?://|doi:|arxiv:|www\.|[a-z0-9._%+\-]+@\S+)", text, re.I):
            continue

        # Skip affiliation lines
        if re.match(
            r"^\d+\s+(department|university|institute|lab|school|college|center|"
            r"google|apple|meta|microsoft|amazon|openai|deepmind|berkeley|mit|stanford)",
            text, re.I,
        ):
            continue

        # Skip footnote symbol lines
        if re.match(r"^[∗†‡§¶✉]\s*\S", text):
            continue

        # Skip very short fragments
        if len(text) < 6:
            continue

        # Figure / table captions
        is_caption = bool(_FIGURE_CAP.match(text))
        if is_caption and not include_captions:
            continue

        # Author-list heuristic: high comma density + digits + no sentence end → skip
        if page < 3:
            comma_ratio = text.count(",") / max(len(text.split()), 1)
            digit_count = sum(c.isdigit() for c in text)
            if comma_ratio > 0.15 and digit_count > 2 and not text.endswith("."):
                continue

        # Classify
        is_section_label = bool(_SECTION_LABEL.match(text))
        is_heading = (
            is_section_label
            or size > body_size + 1.5
            or (is_bold and size >= body_size - 0.5 and len(text) < 120)
        )

        if not title_emitted and (size > body_size + 1 or is_heading):
            blocks.append(Block("title", text, page))
            title_emitted = True
        elif is_heading:
            blocks.append(Block("heading", text, page))
        else:
            blocks.append(Block("body", text, page))
            title_emitted = True

        if max_blocks is not None and len(blocks) >= max_blocks:
            break

    return blocks
