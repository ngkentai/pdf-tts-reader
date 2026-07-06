#!/usr/bin/env python3
"""PDF → TTS pipeline with synchronized HTML viewer.

Usage:
    python pipeline.py paper.pdf [output_dir] [voice] [whisper_model]

Voices (Kokoro American English):
    af_heart  af_sky  am_michael  am_adam
    bf_emma   bm_george  (British)

Whisper models (larger = slower but more accurate):
    tiny  base  small  medium
"""

import gc
import json
import re
import socket
import sys
import tempfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from parser import Block, parse_pdf, extract_references, extract_figure_images

SAMPLE_RATE = 24000  # Kokoro output sample rate


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def run_tts(blocks: list[Block], wav_path: str | Path,
            voice: str = "af_heart", speed: float = 1.0) -> list[tuple[int, int]]:
    """Run Kokoro TTS, streaming each block directly to wav_path.

    Returns exact block_offsets[(start_sample, end_sample), …].
    Caller should cache these to block_offsets.json so assign_timings
    can use accurate time windows instead of estimating.
    Peak memory: Kokoro model + one block's audio only.
    """
    from kokoro import KPipeline

    print("Loading Kokoro TTS model…")
    pipeline = KPipeline(lang_code="a")

    block_offsets: list[tuple[int, int]] = []
    total_samples = 0

    with sf.SoundFile(str(wav_path), "w",
                      samplerate=SAMPLE_RATE, channels=1, subtype="FLOAT") as f:
        for i, block in enumerate(blocks):
            label = f"{block.type[:4]} p{block.page}"
            preview = block.text[:60].replace("\n", " ")
            print(f"  [{i+1}/{len(blocks)}] {label}: {preview}…")

            start = total_samples
            wrote = 0
            for _, _, audio in pipeline(block.tts_text, voice=voice, speed=speed):
                arr = (audio.numpy() if hasattr(audio, "numpy")
                       else np.array(audio)).astype(np.float32)
                f.write(arr)
                wrote += len(arr)

            if wrote == 0:
                silence = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)
                f.write(silence)
                wrote = len(silence)

            total_samples += wrote
            block_offsets.append((start, total_samples))

    return block_offsets


def run_edge_tts(blocks: list[Block], wav_path: str | Path,
                 voice: str = "de-DE-KillianNeural") -> list[tuple[int, int]]:
    """Run Edge TTS (Microsoft, online) for each block, writing audio to wav_path.

    Returns exact block_offsets[(start_sample, end_sample), …].
    Requires internet access. Edge TTS outputs 24 kHz mono MP3.
    """
    import asyncio
    import io
    import edge_tts

    print(f"Connecting to Edge TTS (voice={voice})…")

    async def _synth_one(text: str) -> np.ndarray:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        if not audio_data:
            return np.zeros(SAMPLE_RATE // 4, dtype=np.float32)
        arr, _ = sf.read(io.BytesIO(audio_data))
        return arr.astype(np.float32)

    block_offsets: list[tuple[int, int]] = []
    total_samples = 0

    with sf.SoundFile(str(wav_path), "w",
                      samplerate=SAMPLE_RATE, channels=1, subtype="FLOAT") as f:
        for i, block in enumerate(blocks):
            label = f"{block.type[:4]} p{block.page}"
            preview = block.text[:60].replace("\n", " ")
            print(f"  [{i+1}/{len(blocks)}] {label}: {preview}…")

            start = total_samples
            arr = asyncio.run(_synth_one(block.tts_text))
            f.write(arr)
            total_samples += len(arr)
            block_offsets.append((start, total_samples))

    return block_offsets


# ---------------------------------------------------------------------------
# Whisper word timestamps
# ---------------------------------------------------------------------------

def get_word_timestamps(wav_path: str, model_size: str = "base",
                        chunk_minutes: float = 20.0,
                        language: str | None = "en") -> list[dict]:
    """Transcribe with local Whisper and return [{word, start, end}, …].

    Reads the WAV in chunk_minutes-long slices directly from disk, resampling
    each slice to 16 kHz before passing to Whisper.  Peak memory:
      Whisper model + one resampled chunk (~chunk_minutes × 7.5 MB/min).
    """
    import whisper

    WHISPER_SR = 16000

    info = sf.info(wav_path)
    total_frames = info.frames          # frames at SAMPLE_RATE (24 kHz)
    total_seconds = info.duration

    chunk_frames = int(chunk_minutes * 60 * SAMPLE_RATE)
    n_chunks = max(1, int(np.ceil(total_frames / chunk_frames)))

    print(f"Loading Whisper '{model_size}' model…")
    model = whisper.load_model(model_size)

    all_words: list[dict] = []

    with sf.SoundFile(wav_path) as wav:
        for i in range(n_chunks):
            t0_s = i * chunk_minutes * 60
            start_frame = i * chunk_frames
            frames_to_read = min(chunk_frames, total_frames - start_frame)
            t1_s = t0_s + frames_to_read / SAMPLE_RATE

            if n_chunks > 1:
                print(f"  Transcribing chunk {i+1}/{n_chunks} "
                      f"({t0_s/60:.0f}–{t1_s/60:.0f} min)…")
            else:
                print("Transcribing for word timestamps…")

            wav.seek(start_frame)
            chunk_24k = wav.read(frames_to_read, dtype="float32")

            # Resample 24 kHz → 16 kHz
            n_out = int(len(chunk_24k) * WHISPER_SR / SAMPLE_RATE)
            chunk_16k = np.interp(
                np.linspace(0, len(chunk_24k) - 1, n_out),
                np.arange(len(chunk_24k)),
                chunk_24k,
            ).astype(np.float32)
            del chunk_24k
            gc.collect()

            result = model.transcribe(chunk_16k, word_timestamps=True, language=language)
            del chunk_16k
            gc.collect()

            for seg in result["segments"]:
                for w in seg.get("words", []):
                    all_words.append({
                        "word": w["word"].strip(),
                        "start": float(w["start"]) + t0_s,
                        "end": float(w["end"]) + t0_s,
                    })

    del model
    gc.collect()
    return all_words


# ---------------------------------------------------------------------------
# Timing assignment: map display words → Whisper timestamps
# ---------------------------------------------------------------------------

def _sample_to_whisper_idx(sample_pos: int, whisper_words: list[dict]) -> int:
    """Binary-search for the Whisper word whose start time is closest to sample_pos/SAMPLE_RATE."""
    target_t = sample_pos / SAMPLE_RATE
    lo, hi = 0, len(whisper_words) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if whisper_words[mid]["start"] < target_t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _find_block_boundaries(blocks: list[Block],
                           whisper_words: list[dict],
                           heading_anchors: dict[int, int] | None = None
                           ) -> list[tuple[int, int]]:
    """Assign Whisper word index ranges to blocks using heading-anchored alignment.

    If heading_anchors is provided (block_idx → start_sample from a prior TTS run),
    exact sample positions are converted to Whisper word indices — no text search needed.
    Otherwise heading blocks are located by matching their first distinctive words
    near the proportional estimate.

    Body blocks between consecutive anchors are distributed by TTS word count within
    the anchored window.
    """
    W = len(whisper_words)
    N = len(blocks)
    if N == 0 or W == 0:
        return [(0, 0)] * N

    tts_counts = [max(len(b.tts_text.split()), 1) for b in blocks]
    total_tts = sum(tts_counts)

    cum_tts = [0] * (N + 1)
    for i, c in enumerate(tts_counts):
        cum_tts[i + 1] = cum_tts[i] + c

    def prop_w(block_idx: int) -> int:
        return round(cum_tts[block_idx] / total_tts * W)

    _STRIP_CHARS = re.compile(r"[.,;:!?()\[\]\"'\-]+")
    SKIP = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for",
            "by", "with", "from", "is", "are", "be", "as", "its", "this", "that",
            "their", "they", "we", "it", "all", "has", "have", "been", "can"}

    def norm(w: str) -> str:
        return _STRIP_CHARS.sub("", w).lower()

    whisper_norm = [norm(w["word"]) for w in whisper_words]

    # Pass 1 — find anchors for heading/title blocks
    anchors: dict[int, int] = {0: 0, N: W}

    for i, block in enumerate(blocks):
        if block.type not in ("heading", "title") or i == 0:
            continue

        # Prefer exact sample position from a prior TTS run
        if heading_anchors and i in heading_anchors:
            anchors[i] = _sample_to_whisper_idx(heading_anchors[i], whisper_words)
            continue

        # Fall back to text search
        tokens = block.tts_text.split()
        search_words = [norm(t) for t in tokens
                        if norm(t) not in SKIP and len(norm(t)) > 3][:3]
        if not search_words:
            continue

        est = prop_w(i)
        slack = max(400, int(0.10 * W))
        lo = max(0, est - slack)
        hi = min(W - len(search_words), est + slack)

        best_pos, best_score = est, 0.0
        for j in range(lo, hi):
            score = 0.0
            for k, sw in enumerate(search_words):
                wn = whisper_norm[j + k] if j + k < W else ""
                if wn == sw:
                    score += 1.0
                elif sw and wn and (sw in wn or wn in sw):
                    score += 0.5
            if score > best_score:
                best_score = score
                best_pos = j

        if best_score >= 1.0:
            anchors[i] = best_pos

    # Enforce monotonicity: drop any anchor that would go backward
    valid: list[tuple[int, int]] = [(0, 0)]
    for bi in sorted(anchors.keys())[1:]:
        wi = anchors[bi]
        if wi >= valid[-1][1]:
            valid.append((bi, wi))
    if valid[-1][0] != N:
        valid.append((N, W))
    anchors = dict(valid)

    # Pass 2 — distribute blocks within each anchored segment
    result: list[tuple[int, int]] = [(0, 0)] * N
    sorted_keys = sorted(anchors.keys())

    for seg_i in range(len(sorted_keys) - 1):
        a_start = sorted_keys[seg_i]
        a_end   = sorted_keys[seg_i + 1]
        w_start = anchors[a_start]
        w_end   = anchors[a_end]
        w_range = max(w_end - w_start, 1)

        seg = list(range(a_start, a_end))
        if not seg:
            continue

        seg_tts   = [tts_counts[i] for i in seg]
        seg_total = max(sum(seg_tts), 1)

        w_cursor = w_start
        for j, bi in enumerate(seg):
            n = round(seg_tts[j] / seg_total * w_range)
            result[bi] = (w_cursor, min(w_cursor + n, w_end))
            w_cursor = result[bi][1]
        result[seg[-1]] = (result[seg[-1]][0], w_end)

    # Return boundaries and the validated anchor map (for caller to persist)
    return result, {bi: wi for bi, wi in anchors.items() if bi != N}


def assign_timings(blocks: list[Block], whisper_words: list[dict],
                   block_offsets: list[tuple[int, int]] | None,
                   heading_anchors: dict[int, int] | None = None,
                   ) -> tuple[list[list[dict]], dict[int, int] | None]:
    """For each block, assign a per-display-word timing from Whisper output.

    If block_offsets is provided (exact sample boundaries from the TTS run),
    whisper words are partitioned by time window — this is the most accurate.

    If block_offsets is None (audio pre-existed; no cached boundaries),
    whisper words are partitioned using heading-anchored alignment: heading blocks
    are located in the transcript by word matching, then body blocks are
    distributed proportionally within each anchored section.
    """
    W = len(whisper_words)
    found_anchors: dict[int, int] | None = None

    if W == 0:
        return [[] for _ in blocks], None

    if block_offsets is not None:
        # --- Exact time-window partitioning ---
        def _ww_for_block(start_s, end_s):
            t0, t1 = start_s / SAMPLE_RATE, end_s / SAMPLE_RATE
            return [w for w in whisper_words if t0 <= w["start"] < t1]

        def _fallback_times(start_s, end_s):
            return start_s / SAMPLE_RATE, end_s / SAMPLE_RATE

        block_args = [(s, e) for s, e in block_offsets]
    else:
        # --- Heading-anchored partitioning ---
        block_args, found_anchors = _find_block_boundaries(
            blocks, whisper_words, heading_anchors=heading_anchors
        )

        def _ww_for_block(w_start, w_end):
            return whisper_words[w_start:w_end]

        def _fallback_times(w_start, w_end):
            t0 = whisper_words[w_start]["start"] if w_start < W else 0.0
            t1 = whisper_words[w_end - 1]["end"] if 0 < w_end <= W else t0 + 1.0
            return t0, t1

    per_block: list[list[dict]] = []
    for block, args in zip(blocks, block_args):
        ww = _ww_for_block(*args)
        t_start, t_end = _fallback_times(*args)

        display_words = block.tts_text.split()
        N = len(display_words)
        M = len(ww)

        # Interpolate: map display word i to a fractional position in [0, M],
        # then lerp between consecutive Whisper word starts.  This guarantees
        # strictly monotone, non-overlapping timestamps so the JS binary search
        # never skips a highlighted word.
        def _interp_t(f: float) -> float:
            if M == 0 or f <= 0:
                return t_start
            if f >= M:
                return ww[-1]["end"]
            j = int(f)
            fr = f - j
            if j >= M - 1:
                return ww[-1]["start"] + fr * (ww[-1]["end"] - ww[-1]["start"])
            return ww[j]["start"] + fr * (ww[j + 1]["start"] - ww[j]["start"])

        timings: list[dict] = []
        for i in range(N):
            t_s = _interp_t(i * M / max(N, 1))
            t_e = _interp_t((i + 1) * M / max(N, 1))
            timings.append({"start": t_s, "end": t_e})

        per_block.append(timings)

    return per_block, found_anchors


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

def _build_block_html(text: str, word_idx: int,
                      figures: dict[int, str]) -> tuple[str, int]:
    """Tokenize block text into HTML spans.

    Word boundaries must exactly match tts_text.split() so that the i-th .w
    span aligns with T[i] in the JS timing array.

    _strip_citations uses r"\\s*\\[...\\]" which consumes the leading space before
    a bracket, so "word [ N ]," → tts "word," (1 token).  We replicate that
    behaviour: when a citation regex consumes a leading space, any standalone
    punctuation that directly follows the closing ] is emitted as plain HTML
    (not a .w span) to keep the word count identical.

    - \\s*[N] citation → <span class="ref"> (not counted)
    - Fig N / Figure N → <span class="w figref"> (counted, clickable)
    - everything else  → <span class="w"> (counted)
    """
    # Same leading-\\s* as _strip_citations so we match identical token boundaries
    _CITE = re.compile(r"\s*\[\s*[\d,;\s–—\-]+\s*\]")
    _FIGREF = re.compile(r"\b(Fig(?:ure)?\.?\s*(\d+))\b", re.I)
    _ONLY_PUNCT = re.compile(r"^[,\.;:!\?\)]+$")

    specials: list[tuple[int, int, str, str]] = []

    for m in _CITE.finditer(text):
        nums = ",".join(re.findall(r"\d+", m.group(0)))
        specials.append((m.start(), m.end(), "ref", nums))

    for m in _FIGREF.finditer(text):
        if int(m.group(2)) in figures:
            specials.append((m.start(), m.end(), "fig", m.group(2)))

    specials.sort(key=lambda x: x[0])

    # Remove overlaps (cite may subsume a fig position)
    resolved: list[tuple[int, int, str, str]] = []
    for sp in specials:
        if resolved and sp[0] < resolved[-1][1]:
            continue
        resolved.append(sp)
    specials = resolved

    parts: list[str] = []
    pos = 0
    merge_next_punct = False  # True when last cite consumed its leading space

    def emit_chunk(chunk: str) -> None:
        nonlocal word_idx, merge_next_punct
        for tok in re.findall(r"\S+|\s+", chunk):
            if tok.strip():
                if merge_next_punct and _ONLY_PUNCT.match(tok):
                    parts.append(_esc(tok))   # plain text — merged with prev word in tts
                else:
                    parts.append(f'<span class="w" data-i="{word_idx}">{_esc(tok)}</span>')
                    word_idx += 1
                merge_next_punct = False
            else:
                merge_next_punct = False      # whitespace → next token is a fresh word
                parts.append(tok)

    for s_start, s_end, kind, data in specials:
        if s_start < pos:
            continue
        emit_chunk(text[pos:s_start])
        raw = text[s_start:s_end]

        if kind == "ref":
            display_raw = raw.lstrip()
            had_leading_space = len(raw) != len(display_raw)
            if had_leading_space:
                parts.append(" ")        # emit the space that was inside the match
                merge_next_punct = True  # next standalone punct is merged in tts
            parts.append(
                f'<span class="ref" data-n="{_esc(data)}">{_esc(display_raw)}</span>'
            )
        elif kind == "fig":
            parts.append(
                f'<span class="w figref" data-i="{word_idx}" '
                f'data-fig="{data}">{_esc(raw)}</span>'
            )
            word_idx += 1
            merge_next_punct = False

        pos = s_end

    emit_chunk(text[pos:])
    return "".join(parts), word_idx


def generate_html(blocks: list[Block], per_block_timings: list[list[dict]],
                  audio_filename: str, output_path: Path,
                  refs: dict[int, str] | None = None,
                  figures: dict[int, str] | None = None,
                  embed_audio_path: Path | None = None,
                  lang: str = "en",
                  doc_id: str = ""):
    """Write the synchronized HTML viewer.

    If embed_audio_path is given, the audio file is base64-encoded and embedded
    as a data URI — the resulting HTML is fully self-contained and works when
    opened directly from the filesystem (e.g. iPhone Files app, iOS Safari).
    """
    refs = refs or {}
    figures = figures or {}

    # Flatten timings and collect chapter metadata
    all_timings: list[dict] = []
    chapters: list[dict] = []

    for block, timings in zip(blocks, per_block_timings):
        if block.type in ("title", "heading") and timings:
            chapters.append({"title": block.tts_text[:70], "t": round(timings[0]["start"], 3)})
        all_timings.extend(timings)

    timings_json = json.dumps(
        [{"s": round(t["start"], 3), "e": round(t["end"], 3)} for t in all_timings]
    )
    chapters_json = json.dumps(chapters)
    refs_json = json.dumps({str(k): v for k, v in refs.items()})
    # figures are embedded as data-URIs directly in the HTML spans
    figures_json = json.dumps({
        str(k): f"data:image/png;base64,{v}" for k, v in figures.items()
    })

    if embed_audio_path is not None:
        import base64
        audio_b64 = base64.b64encode(embed_audio_path.read_bytes()).decode()
        audio_mime = "audio/mpeg" if str(embed_audio_path).endswith(".mp3") else "audio/wav"
        audio_src = f"data:{audio_mime};base64,{audio_b64}"
        audio_type = audio_mime
    else:
        audio_src = audio_filename
        audio_type = "audio/wav" if audio_filename.endswith(".wav") else "audio/mpeg"

    # Build content HTML
    word_idx = 0
    content_parts: list[str] = []
    ch_count = 0
    for block, timings in zip(blocks, per_block_timings):
        inner, word_idx = _build_block_html(block.text, word_idx, figures)
        if block.type == "title":
            content_parts.append(f'<h1 id="ch-{ch_count}">{inner}</h1>')
            ch_count += 1
        elif block.type == "heading":
            content_parts.append(f'<h2 id="ch-{ch_count}">{inner}</h2>')
            ch_count += 1
        else:
            content_parts.append(f"<p>{inner}</p>")

    content_html = "\n".join(content_parts)
    total_words = word_idx

    html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
  <title>PDF Reader</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    html{{-webkit-text-size-adjust:100%}}
    body{{font-family:Georgia,'Times New Roman',serif;font-size:19px;line-height:1.8;
         color:#111;background:#faf9f7;padding-bottom:120px}}

    /* ── Player bar ── */
    #bar{{position:sticky;top:0;z-index:100;background:#fff;
          border-bottom:1px solid #ddd;padding:8px 12px 6px;
          display:flex;flex-direction:column;gap:5px;
          box-shadow:0 1px 6px rgba(0,0,0,.08)}}
    #controls{{display:flex;align-items:center;gap:8px;font-family:-apple-system,sans-serif}}
    #btn-play{{font-size:22px;background:none;border:none;cursor:pointer;padding:0 2px;line-height:1;color:#222}}
    #btn-back{{font-size:14px;background:none;border:none;cursor:pointer;padding:0 2px;color:#555}}
    #time{{font-size:12px;color:#777;flex:1;text-align:right}}
    #speed{{font-size:12px;padding:2px 4px;border:1px solid #ccc;border-radius:4px;background:#fff;cursor:pointer}}
    #progress-wrap{{position:relative;height:20px;cursor:pointer}}
    #prog-track{{position:absolute;top:50%;transform:translateY(-50%);left:0;right:0;height:4px;background:#ddd;border-radius:2px}}
    #prog-fill{{height:100%;background:#e8a020;border-radius:2px;width:0%;transition:width 0.25s linear;pointer-events:none}}
    .ch-mark{{position:absolute;top:50%;transform:translate(-50%,-50%);width:3px;height:12px;background:#888;border-radius:1px;cursor:pointer;z-index:2}}
    .ch-mark:hover::after{{content:attr(data-title);position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
      background:#333;color:#fff;font-size:10px;font-family:-apple-system,sans-serif;white-space:nowrap;
      padding:2px 6px;border-radius:3px;pointer-events:none;max-width:220px;overflow:hidden;text-overflow:ellipsis}}
    #winfo{{font-family:-apple-system,sans-serif;font-size:11px;color:#aaa;text-align:right;padding-right:2px}}

    /* ── Content ── */
    #content{{max-width:700px;margin:0 auto;padding:20px 16px 60px}}
    h1{{font-size:1.45em;margin:0.6em 0 0.5em;line-height:1.3}}
    h2{{font-size:1.15em;color:#2c2c2c;margin:1.6em 0 0.35em;border-bottom:1px solid #e8e8e8;padding-bottom:3px}}
    p{{margin-bottom:1.1em}}
    .w{{border-radius:3px;padding:0 1px;cursor:pointer;transition:background 0.07s,color 0.07s}}
    .w:hover{{background:#f0e8aa}}
    .w.active{{background:#ffe566;color:#000}}
    .w.past{{color:#bbb}}

    /* ── Citation refs ── */
    .ref{{color:#1a6fc4;cursor:pointer;font-size:0.78em;vertical-align:super;
          white-space:nowrap;display:inline-block;padding:0 1px}}
    .ref:hover{{text-decoration:underline}}

    /* ── Figure refs ── */
    .figref{{color:#1a6fc4;cursor:pointer;border-bottom:1px dotted #1a6fc4}}
    .figref:hover{{background:#e8f0ff}}
    .figref.active{{background:#ffe566;color:#000;border-color:transparent}}
    .figref.past{{color:#bbb;border-color:#bbb}}

    /* ── Modal overlay ── */
    #modal{{display:none;position:fixed;inset:0;z-index:500;
            background:rgba(0,0,0,.55);align-items:center;justify-content:center;padding:16px}}
    #modal.open{{display:flex}}
    #modal-box{{background:#fff;border-radius:10px;max-width:680px;width:100%;
                max-height:90vh;overflow-y:auto;padding:20px;position:relative;
                font-family:-apple-system,sans-serif}}
    #modal-close{{position:absolute;top:10px;right:14px;font-size:22px;
                  background:none;border:none;cursor:pointer;color:#555;line-height:1}}
    #modal-title{{font-size:14px;font-weight:600;color:#444;margin-bottom:10px}}
    #modal-body{{font-size:14px;line-height:1.6;color:#222}}
    #modal-body img{{max-width:100%;border-radius:4px;margin-bottom:8px}}
    #modal-body p{{margin:0}}

    @media(prefers-color-scheme:dark){{
      body{{background:#1c1c1e;color:#e5e5ea}}
      #bar{{background:#1c1c1e;border-color:#3a3a3c;box-shadow:none}}
      #btn-play{{color:#e5e5ea}} #btn-back{{color:#aaa}}
      #speed{{background:#2c2c2e;border-color:#48484a;color:#e5e5ea}}
      #prog-track{{background:#3a3a3c}} #prog-fill{{background:#c8901a}}
      .ch-mark{{background:#888}}
      h2{{color:#c7c7cc;border-color:#3a3a3c}}
      .w:hover{{background:#3a3a3c}}
      .w.active{{background:#b8860b;color:#fff}}
      .w.past{{color:#555}}
      .ref{{color:#5aa4f0}}
      .figref{{color:#5aa4f0;border-color:#5aa4f0}}
      .figref:hover{{background:#1e2a3a}}
      #modal-box{{background:#2c2c2e;color:#e5e5ea}}
      #modal-close{{color:#aaa}}
      #modal-title{{color:#aaa}}
      #modal-body{{color:#e5e5ea}}
    }}
  </style>
</head>
<body>
<div id="bar">
  <audio id="player" preload="auto">
    <source src="{audio_src}" type="{audio_type}">
  </audio>
  <div id="controls">
    <button id="btn-play" title="Play/Pause">▶</button>
    <button id="btn-back" title="Back 10s">↺10s</button>
    <span id="time">0:00 / 0:00</span>
    <select id="speed" title="Playback speed">
      <option value="0.8">0.8×</option>
      <option value="1" selected>1×</option>
      <option value="1.3">1.3×</option>
      <option value="1.6">1.6×</option>
    </select>
  </div>
  <div id="progress-wrap">
    <div id="prog-track"><div id="prog-fill"></div></div>
  </div>
  <div id="winfo">Word 0 / {total_words}</div>
</div>

<!-- Reference / Figure modal -->
<div id="modal">
  <div id="modal-box">
    <button id="modal-close" title="Close">✕</button>
    <div id="modal-title"></div>
    <div id="modal-body"></div>
  </div>
</div>

<div id="content">
{content_html}
</div>
<script>
const T={timings_json};
const CHAPTERS={chapters_json};
const REFS={refs_json};
const FIGS={figures_json};

const player=document.getElementById('player');
const btnPlay=document.getElementById('btn-play');
const btnBack=document.getElementById('btn-back');
const timeEl=document.getElementById('time');
const speedSel=document.getElementById('speed');
const progFill=document.getElementById('prog-fill');
const progWrap=document.getElementById('progress-wrap');
const winfoEl=document.getElementById('winfo');
const spans=document.querySelectorAll('.w');
const modal=document.getElementById('modal');
const modalTitle=document.getElementById('modal-title');
const modalBody=document.getElementById('modal-body');
let cur=-1,scrollLock=false;

/* ── Resume / save position ── */
const DOC_ID='{doc_id}';
const LS_KEY='tts_pos_'+DOC_ID;
let lsSaveAt=0;
(function(){{
  if(!DOC_ID)return;
  const saved=parseFloat(localStorage.getItem(LS_KEY)||'0');
  if(saved>30){{
    player.addEventListener('canplay',function h(){{
      player.removeEventListener('canplay',h);
      player.currentTime=saved;
    }});
  }}
}})();

/* ── Binary search ── */
function bs(t){{
  let lo=0,hi=T.length-1;
  while(lo<=hi){{
    const m=(lo+hi)>>1;
    if(T[m].e<=t)lo=m+1;
    else if(T[m].s>t)hi=m-1;
    else return m;
  }}
  return -1;
}}
function fmt(s){{
  if(!isFinite(s))return'0:00';
  const m=Math.floor(s/60),sc=Math.floor(s)%60;
  return m+':'+(sc<10?'0':'')+sc;
}}

/* ── Playback controls ── */
btnPlay.addEventListener('click',()=>{{
  if(player.paused){{player.play();btnPlay.textContent='⏸';}}
  else{{player.pause();btnPlay.textContent='▶';}}
}});
player.addEventListener('ended',()=>btnPlay.textContent='▶');
btnBack.addEventListener('click',()=>player.currentTime=Math.max(0,player.currentTime-10));
speedSel.addEventListener('change',()=>player.playbackRate=parseFloat(speedSel.value));

/* ── Chapter markers ──
   The audio metadata often finishes loading while the browser is still
   parsing the (large) content HTML above this script, i.e. before the
   loadedmetadata listener exists — so also render immediately if ready. */
function renderChapters(){{
  const dur=player.duration;
  if(!isFinite(dur)||dur<=0||progWrap.querySelector('.ch-mark'))return;
  CHAPTERS.forEach(ch=>{{
    const mk=document.createElement('div');
    mk.className='ch-mark';
    mk.style.left=(ch.t/dur*100)+'%';
    mk.setAttribute('data-title',ch.title);
    mk.addEventListener('click',e=>{{
      e.stopPropagation();
      player.currentTime=ch.t;
      if(player.paused)player.play();
    }});
    progWrap.appendChild(mk);
  }});
}}
player.addEventListener('loadedmetadata',renderChapters);
player.addEventListener('durationchange',renderChapters);
if(player.readyState>=1)renderChapters();

/* ── Progress bar scrubbing ── */
function seekTo(e){{
  const r=progWrap.getBoundingClientRect();
  player.currentTime=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*(player.duration||0);
}}
progWrap.addEventListener('click',seekTo);
progWrap.addEventListener('touchend',e=>{{seekTo(e.changedTouches[0]);e.preventDefault();}},{{passive:false}});

/* ── Time update + word highlight ── */
player.addEventListener('timeupdate',()=>{{
  const t=player.currentTime,dur=player.duration||1;
  timeEl.textContent=fmt(t)+' / '+fmt(dur);
  progFill.style.width=(t/dur*100)+'%';
  if(DOC_ID){{const now=Date.now();if(now-lsSaveAt>5000&&t>1){{lsSaveAt=now;try{{localStorage.setItem(LS_KEY,t.toFixed(1));}}catch(_){{}}}}}}

  const idx=bs(t);
  if(idx===cur)return;
  if(cur>=0&&cur<spans.length){{spans[cur].classList.remove('active');spans[cur].classList.add('past');}}
  cur=idx;
  if(idx>=0&&idx<spans.length){{
    spans[idx].classList.add('active');
    if(!scrollLock)spans[idx].scrollIntoView({{behavior:'smooth',block:'center'}});
    winfoEl.textContent='Word '+(idx+1)+' / {total_words}';
  }}
}});
player.addEventListener('seeked',()=>{{
  const t=player.currentTime;
  spans.forEach((s,i)=>{{s.classList.remove('active','past');if(i<T.length&&T[i].e<t)s.classList.add('past');}});
  cur=-1;
}});

/* ── Tap word → seek ── */
spans.forEach((span,i)=>{{
  if(span.classList.contains('figref'))return; // figref has its own handler
  span.addEventListener('click',()=>{{
    if(i<T.length){{player.currentTime=T[i].s;player.play();btnPlay.textContent='⏸';}}
  }});
}});

/* ── Manual scroll lock ── */
let scrollTimer;
window.addEventListener('scroll',()=>{{
  scrollLock=true;clearTimeout(scrollTimer);
  scrollTimer=setTimeout(()=>scrollLock=false,3000);
}},{{passive:true}});

/* ── Modal helpers ── */
function openModal(title,bodyHtml){{
  modalTitle.textContent=title;
  modalBody.innerHTML=bodyHtml;
  modal.classList.add('open');
}}
function closeModal(){{modal.classList.remove('open');}}
document.getElementById('modal-close').addEventListener('click',closeModal);
modal.addEventListener('click',e=>{{if(e.target===modal)closeModal();}});
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeModal();}});

/* ── Citation refs ── */
document.querySelectorAll('.ref').forEach(el=>{{
  el.addEventListener('click',e=>{{
    e.stopPropagation();
    const nums=el.dataset.n.split(',').map(s=>s.trim());
    const lines=nums.map(n=>{{
      const txt=REFS[n];
      return txt?`<p><strong>[`+n+`]</strong> `+txt+`</p>`:`<p><strong>[`+n+`]</strong> <em>Reference not found.</em></p>`;
    }});
    openModal('Reference'+( nums.length>1?'s':''),lines.join('<br>'));
  }});
}});

/* ── Figure refs ── */
document.querySelectorAll('.figref').forEach(el=>{{
  el.addEventListener('click',e=>{{
    e.stopPropagation();
    const n=el.dataset.fig;
    const src=FIGS[n];
    if(!src)return;
    openModal('Figure '+n,`<img src="`+src+`" alt="Figure `+n+`"><p style="color:#888;font-size:12px;margin-top:6px">Figure `+n+`</p>`);
  }});
}});
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# MP3 export
# ---------------------------------------------------------------------------

def _write_mp3(wav_path: Path, mp3_path: Path, bitrate: int = 128) -> None:
    """Convert WAV to MP3 using lameenc, streaming 30-second chunks.

    Reads float32 WAV, converts to int16 PCM, encodes with lameenc.
    Peak memory: one 30-second chunk (~5 MB) plus the encoder.
    """
    import lameenc

    info = sf.info(str(wav_path))
    total_frames = info.frames

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)

    CHUNK = SAMPLE_RATE * 30  # 30 s
    done = 0

    with sf.SoundFile(str(wav_path)) as wav_f:
        with open(str(mp3_path), "wb") as mp3_f:
            while True:
                chunk = wav_f.read(CHUNK, dtype="float32")
                if len(chunk) == 0:
                    break
                pcm = (chunk * 32767).astype(np.int16)
                mp3_f.write(encoder.encode(pcm.tobytes()))
                done += len(chunk)
                print(f"\r  Encoding MP3… {done / total_frames * 100:.0f}%",
                      end="", flush=True)
            mp3_f.write(encoder.flush())
    print()


# ---------------------------------------------------------------------------
# Manifest (landing page index)
# ---------------------------------------------------------------------------

def _update_manifest(output_dir: Path, blocks: list[Block],
                     duration_min: float, language: str) -> None:
    """Create or update manifest.json in the parent directory for the landing page."""
    import datetime

    manifest_path = output_dir.parent / "manifest.json"

    title = output_dir.name.replace("_tts", "").replace("_", " ").title()
    # search early blocks first, then remaining headings/titles
    early = blocks[:20]
    rest_headings = [b for b in blocks[20:] if b.type in ("title", "heading")]
    for b in early + rest_headings:
        raw = b.text.strip()
        # split on newlines and on common PDF merge artefacts (replacement char, 2+ spaces)
        parts = re.split(r"\n|\s{2,}|�|•|", raw)
        for part in parts:
            t = part.strip()[:120]
            if not (10 < len(t) < 120):
                continue
            if not t[0].isalpha():
                continue
            if re.search(r'\b(Seite|Page)\s+\d|\d+\s+(of|von)\s+\d+', t, re.I):
                continue
            letter_ratio = sum(1 for c in t if c.isalpha()) / len(t)
            if letter_ratio < 0.55:
                continue
            title = t
            break
        else:
            continue
        break

    entry = {
        "id": output_dir.name,
        "title": title,
        "language": language if language != "auto" else "und",
        "duration_min": round(duration_min, 1),
        "viewer": f"{output_dir.name}/viewer.html",
        "generated": datetime.date.today().isoformat(),
    }

    # Audio file + sizes so the iOS app knows what to download and how big it is.
    # The viewer's <source> tag is the source of truth (skipped for embedded data URIs).
    viewer_path = output_dir / "viewer.html"
    if viewer_path.exists():
        entry["viewer_bytes"] = viewer_path.stat().st_size
        with open(viewer_path, encoding="utf-8") as f:
            head = f.read(200_000)
        m = re.search(r'<source src="([^"]+)"', head)
        if m and not m.group(1).startswith("data:"):
            audio_name = m.group(1)
            audio_path = output_dir / audio_name
            if audio_path.exists():
                entry["audio"] = f"{output_dir.name}/{audio_name}"
                entry["audio_bytes"] = audio_path.stat().st_size

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"documents": []}

    ids = [d["id"] for d in manifest["documents"]]
    if entry["id"] in ids:
        manifest["documents"][ids.index(entry["id"])] = entry
    else:
        manifest["documents"].append(entry)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"  Updated {manifest_path.name} ({len(manifest['documents'])} document(s)).")


# ---------------------------------------------------------------------------
# QR code helper
# ---------------------------------------------------------------------------

def print_qr(url: str):
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        print(f"\n  Scan to open on phone: {url}\n")
    except ImportError:
        print(f"\n  Open on phone: {url}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="PDF → TTS synchronized viewer")
    ap.add_argument("pdf", help="Input PDF file")
    ap.add_argument("output_dir", nargs="?", help="Output directory (default: <pdf>_tts/)")
    ap.add_argument("--voice", default="af_heart",
                    help="Kokoro voice (default: af_heart). Ignored when --tts-engine=edge.")
    ap.add_argument("--tts-engine", choices=["kokoro", "edge"], default="kokoro",
                    help="TTS engine: kokoro (local, English only) or edge (online, multilingual). Default: kokoro.")
    ap.add_argument("--edge-voice", default="de-DE-KillianNeural",
                    help="Edge TTS voice name (default: de-DE-KillianNeural). Used when --tts-engine=edge.")
    ap.add_argument("--language", default="en",
                    help="Language code for Whisper transcription (default: en). E.g. de, fr, ja. Use 'auto' to let Whisper detect.")
    ap.add_argument("--whisper-model", default="base",
                    help="Whisper model size: tiny/base/small/medium (default: base)")
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="Limit to first N blocks (for quick tests)")
    ap.add_argument("--chunk-minutes", type=float, default=20.0,
                    help="Whisper chunk size in minutes (default: 20). Reduce if OOM.")
    ap.add_argument("--mp3-bitrate", type=int, default=128,
                    help="MP3 bitrate in kbps (default: 128). Use 32–64 with --embed-audio.")
    ap.add_argument("--embed-audio", action="store_true",
                    help="Embed audio as base64 in viewer.html (self-contained, works on iPhone).")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_dir) if args.output_dir else pdf_path.parent / (pdf_path.stem + "_tts")
    voice = args.voice
    tts_engine = args.tts_engine
    edge_voice = args.edge_voice
    whisper_language = None if args.language == "auto" else args.language
    whisper_model = args.whisper_model
    chunk_minutes = args.chunk_minutes
    mp3_bitrate = args.mp3_bitrate
    embed_audio = args.embed_audio

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path  = output_dir / "audio.wav"
    mp3_path  = output_dir / f"audio_{mp3_bitrate}k.mp3" if mp3_bitrate != 128 else output_dir / "audio.mp3"
    html_path = output_dir / "viewer.html"

    # --- Parse PDF ---
    print(f"\nParsing {pdf_path.name}…")
    blocks = parse_pdf(str(pdf_path), max_blocks=args.max_blocks)
    if not blocks:
        print("ERROR: no readable text found in PDF.")
        sys.exit(1)

    total_words = sum(len(b.tts_text.split()) for b in blocks)
    est_min = total_words / 150
    est_wav_mb = est_min * 60 * SAMPLE_RATE * 4 / 1e6  # float32 bytes
    print(f"  {len(blocks)} blocks, ~{total_words} words")
    print(f"  Estimated audio: ~{est_min:.1f} min  |  WAV: ~{est_wav_mb:.0f} MB")

    # Memory estimate and warning
    avail_gb = _available_ram_gb()
    _WHISPER_MODEL_MB = {"tiny": 39, "base": 74, "small": 244, "medium": 769}
    model_mb = _WHISPER_MODEL_MB.get(whisper_model, 200)
    chunk_mb = chunk_minutes * 60 * 16000 * 4 / 1e6   # resampled chunk at 16 kHz
    peak_whisper_mb = model_mb + chunk_mb + 300        # 300 MB PyTorch overhead
    if avail_gb:
        flag = "⚠ LOW" if peak_whisper_mb / 1e3 > avail_gb * 0.7 else "OK"
        print(f"  Available RAM: {avail_gb:.1f} GB  |  "
              f"Whisper peak estimate: ~{peak_whisper_mb/1e3:.1f} GB  [{flag}]")
        if flag == "⚠ LOW":
            print(f"  Tip: use --whisper-model tiny or --chunk-minutes 10 to reduce RAM")

    # --- TTS ---
    offsets_path = output_dir / "block_offsets.json"
    if wav_path.exists():
        print(f"\nAudio already exists ({wav_path}), skipping TTS.")
        if offsets_path.exists():
            block_offsets = [tuple(x) for x in json.loads(offsets_path.read_text())]
            print(f"  Loaded exact block offsets from cache.")
        else:
            block_offsets = None   # assign_timings will use word-count partition
            print(f"  No cached block offsets — will align by heading-anchored proportioning.")
    else:
        if tts_engine == "edge":
            block_offsets = run_edge_tts(blocks, wav_path=wav_path, voice=edge_voice)
        else:
            print(f"\nRunning TTS (voice={voice})…")
            block_offsets = run_tts(blocks, wav_path=wav_path, voice=voice)
        offsets_path.write_text(json.dumps(block_offsets))
        print(f"  Saved exact block offsets to {offsets_path.name}.")

    # Save heading_anchors.json — start samples for each heading/title block.
    # These let future runs place anchors exactly without text-searching the transcript.
    anchors_path = output_dir / "heading_anchors.json"
    if not anchors_path.exists() and block_offsets is not None:
        ha = {i: block_offsets[i][0]
              for i, b in enumerate(blocks)
              if b.type in ("heading", "title")}
        anchors_path.write_text(json.dumps(ha))
        print(f"  Saved {len(ha)} heading anchors to {anchors_path.name}.")

    # --- MP3 ---
    if not mp3_path.exists():
        print(f"\nConverting WAV → MP3 ({mp3_bitrate} kbps)…")
        _write_mp3(wav_path, mp3_path, bitrate=mp3_bitrate)
        print(f"  Saved {mp3_path.name} ({mp3_path.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"\nMP3 already exists ({mp3_path.name}), skipping conversion.")

    audio_filename = mp3_path.name if mp3_path.exists() else wav_path.name

    if embed_audio:
        mp3_mb = mp3_path.stat().st_size / 1e6 if mp3_path.exists() else 0
        html_mb = mp3_mb * 1.37  # base64 overhead
        if html_mb > 200:
            print(f"\n⚠  Embedding {mp3_mb:.0f} MB audio → ~{html_mb:.0f} MB HTML.")
            print(f"   Consider --mp3-bitrate 32 for voice content (re-delete audio.mp3 first).")
        else:
            print(f"\nEmbedding audio ({mp3_mb:.0f} MB → ~{html_mb:.0f} MB HTML)…")

    # --- Whisper ---
    timestamps_path = output_dir / "timestamps.json"
    if timestamps_path.exists():
        print(f"\nLoading cached timestamps from {timestamps_path}…")
        whisper_words = json.loads(timestamps_path.read_text())
    else:
        print(f"\nRunning Whisper (model={whisper_model}, chunk={chunk_minutes:.0f} min)…")
        whisper_words = get_word_timestamps(str(wav_path), model_size=whisper_model,
                                            chunk_minutes=chunk_minutes,
                                            language=whisper_language)
        timestamps_path.write_text(json.dumps(whisper_words, indent=2))
        print(f"  {len(whisper_words)} word timestamps saved.")

    # Load heading anchors for the fallback (no block_offsets) case
    heading_anchors: dict[int, int] | None = None
    if block_offsets is None and anchors_path.exists():
        raw = json.loads(anchors_path.read_text())
        heading_anchors = {int(k): v for k, v in raw.items()}
        print(f"  Using {len(heading_anchors)} saved heading anchors for alignment.")

    # --- Assign timings to display words ---
    per_block_timings, found_anchors = assign_timings(
        blocks, whisper_words, block_offsets, heading_anchors=heading_anchors
    )

    # Persist heading anchors discovered by text search so future runs skip the search
    if not anchors_path.exists() and found_anchors:
        # Convert Whisper word indices → sample positions using Whisper timestamps
        ha_samples = {
            i: int(whisper_words[wi]["start"] * SAMPLE_RATE)
            for i, wi in found_anchors.items()
            if i > 0 and wi < len(whisper_words)
        }
        anchors_path.write_text(json.dumps(ha_samples))
        print(f"  Saved {len(ha_samples)} heading anchors to {anchors_path.name}.")

    # --- Extract references and figures (fast, no reprocessing needed) ---
    print(f"\nExtracting references and figures from PDF…")
    refs = extract_references(str(pdf_path))
    figures = extract_figure_images(str(pdf_path))
    print(f"  {len(refs)} references, {len(figures)} figures extracted")

    # --- Generate HTML ---
    print(f"Generating HTML viewer…")
    embed_path = mp3_path if embed_audio and mp3_path.exists() else None
    html_lang = args.language if args.language != "auto" else "und"
    generate_html(blocks, per_block_timings, audio_filename, html_path,
                  refs=refs, figures=figures, embed_audio_path=embed_path,
                  lang=html_lang, doc_id=output_dir.name)

    # Update manifest.json for the landing page
    wav_info = sf.info(str(wav_path))
    _update_manifest(output_dir, blocks,
                     duration_min=wav_info.duration / 60,
                     language=args.language)

    # --- Summary ---
    dur = sf.info(str(wav_path)).duration
    print(f"\n{'─'*50}")
    print(f"  WAV   : {wav_path}  ({dur/60:.1f} min)")
    if mp3_path.exists():
        print(f"  MP3   : {mp3_path}  ({mp3_path.stat().st_size / 1e6:.0f} MB)")
    html_mb = html_path.stat().st_size / 1e6
    embedded_note = "  ← self-contained, copy to iPhone" if embed_audio else ""
    print(f"  Viewer: {html_path}  ({html_mb:.0f} MB){embedded_note}")
    print(f"{'─'*50}")
    print(f"\nServe locally for phone access:")
    print(f"  cd \"{output_dir}\" && python3 -m http.server 8080\n")

    # Try to print QR code with local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print_qr(f"http://{local_ip}:8080/viewer.html")
    except Exception:
        pass


def _available_ram_gb() -> float | None:
    """Return available system RAM in GB, or None if undetectable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return None


def _estimate_offsets(blocks: list[Block], total_samples: int) -> list[tuple[int, int]]:
    """Estimate per-block audio offsets by word-count proportion."""
    word_counts = [max(len(b.tts_text.split()), 1) for b in blocks]
    total = sum(word_counts)
    offsets = []
    cursor = 0
    for wc in word_counts:
        samples = round(wc / total * total_samples)
        offsets.append((cursor, cursor + samples))
        cursor += samples
    # fix last block to hit exactly total_samples
    if offsets:
        offsets[-1] = (offsets[-1][0], total_samples)
    return offsets


if __name__ == "__main__":
    main()
