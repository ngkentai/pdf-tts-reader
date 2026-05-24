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

def run_tts(blocks: list[Block], voice: str = "af_heart", speed: float = 1.0):
    """Run Kokoro TTS on every block. Returns (full_audio_array, block_offsets).

    block_offsets[i] = (start_sample, end_sample) for blocks[i].
    """
    from kokoro import KPipeline

    print("Loading Kokoro TTS model…")
    pipeline = KPipeline(lang_code="a")

    audio_chunks: list[np.ndarray] = []
    block_offsets: list[tuple[int, int]] = []
    total_samples = 0

    for i, block in enumerate(blocks):
        label = f"{block.type[:4]} p{block.page}"
        preview = block.text[:60].replace("\n", " ")
        print(f"  [{i+1}/{len(blocks)}] {label}: {preview}…")

        chunk_parts: list[np.ndarray] = []
        for _, _, audio in pipeline(block.tts_text, voice=voice, speed=speed):
            arr = audio.numpy() if hasattr(audio, "numpy") else np.array(audio)
            chunk_parts.append(arr.astype(np.float32))

        if chunk_parts:
            block_audio = np.concatenate(chunk_parts)
        else:
            block_audio = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)  # 250 ms silence

        start = total_samples
        end = total_samples + len(block_audio)
        block_offsets.append((start, end))
        audio_chunks.append(block_audio)
        total_samples = end

    full_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
    return full_audio, block_offsets


# ---------------------------------------------------------------------------
# Whisper word timestamps
# ---------------------------------------------------------------------------

def get_word_timestamps(audio: np.ndarray, model_size: str = "base") -> list[dict]:
    """Transcribe with local Whisper and return [{word, start, end}, …].

    Accepts a numpy audio array at SAMPLE_RATE.  Resamples to Whisper's
    16 kHz internally — no ffmpeg required.
    """
    import whisper

    WHISPER_SR = 16000

    # Resample to 16 kHz using linear interpolation (no scipy/ffmpeg needed)
    if SAMPLE_RATE != WHISPER_SR:
        n_out = int(len(audio) * WHISPER_SR / SAMPLE_RATE)
        audio_16k = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    else:
        audio_16k = audio.astype(np.float32)

    print(f"Loading Whisper '{model_size}' model…")
    model = whisper.load_model(model_size)

    print("Transcribing for word timestamps…")
    result = model.transcribe(audio_16k, word_timestamps=True, language="en")

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
    return words


# ---------------------------------------------------------------------------
# Timing assignment: map display words → Whisper timestamps
# ---------------------------------------------------------------------------

def assign_timings(blocks: list[Block], whisper_words: list[dict],
                   block_offsets: list[tuple[int, int]]) -> list[list[dict]]:
    """For each block, assign a per-display-word timing from Whisper output.

    Strategy: within each block's known audio time window, find the Whisper
    words that fall there and map them positionally to the display words.
    This is approximate but robust to TTS number expansion etc.
    """
    total_samples = block_offsets[-1][1] if block_offsets else 1

    per_block: list[list[dict]] = []
    for block, (start_s, end_s) in zip(blocks, block_offsets):
        t_start = start_s / SAMPLE_RATE
        t_end = end_s / SAMPLE_RATE

        # Whisper words that fall in this block's time window
        ww = [w for w in whisper_words if t_start <= w["start"] < t_end]

        # Use tts_text word count — matches what was actually spoken and
        # what the HTML will emit as .w spans (citations stripped from both)
        display_words = block.tts_text.split()
        N = len(display_words)
        M = len(ww)

        timings: list[dict] = []
        for i in range(N):
            if M == 0:
                # Interpolate linearly across block duration
                frac = i / max(N - 1, 1)
                s = t_start + frac * (t_end - t_start)
                e = s + (t_end - t_start) / max(N, 1)
                timings.append({"start": s, "end": min(e, t_end)})
            else:
                j = round(i / max(N - 1, 1) * (M - 1)) if N > 1 else 0
                j = max(0, min(j, M - 1))
                timings.append({"start": ww[j]["start"], "end": ww[j]["end"]})

        per_block.append(timings)

    return per_block


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

def _build_block_html(text: str, word_idx: int,
                      figures: dict[int, str]) -> tuple[str, int]:
    """Tokenize block text into HTML spans.

    - [N] / [N,M] citation tokens → <span class="ref"> (not a .w span, not counted)
    - Fig N / Figure N text       → <span class="w figref"> (counted, clickable)
    - all other non-space tokens  → <span class="w"> (counted)
    """
    # Find special regions: citations and figure references
    specials: list[tuple[int, int, str, str]] = []  # (start, end, kind, data)

    for m in re.finditer(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", text):
        specials.append((m.start(), m.end(), "ref", m.group(1)))

    for m in re.finditer(r"\b(Fig(?:ure)?\.?\s*(\d+))\b", text, re.I):
        # Only mark as figref if we actually have that figure image
        if int(m.group(2)) in figures:
            specials.append((m.start(), m.end(), "fig", m.group(2)))

    specials.sort(key=lambda x: x[0])

    parts: list[str] = []
    pos = 0

    def emit_plain(chunk: str) -> None:
        nonlocal word_idx
        for tok in re.findall(r"\S+|\s+", chunk):
            if tok.strip():
                parts.append(
                    f'<span class="w" data-i="{word_idx}">{_esc(tok)}</span>'
                )
                word_idx += 1
            else:
                parts.append(tok)

    for s_start, s_end, kind, data in specials:
        if s_start < pos:          # already consumed (overlap guard)
            continue
        emit_plain(text[pos:s_start])
        raw = text[s_start:s_end]
        if kind == "ref":
            nums = re.sub(r"\s", "", data)   # "1,2" normalised
            parts.append(
                f'<span class="ref" data-n="{_esc(nums)}">{_esc(raw)}</span>'
            )
        elif kind == "fig":
            parts.append(
                f'<span class="w figref" data-i="{word_idx}" '
                f'data-fig="{data}">{_esc(raw)}</span>'
            )
            word_idx += 1        # figref words ARE counted (they're spoken)
        pos = s_end

    emit_plain(text[pos:])
    return "".join(parts), word_idx


def generate_html(blocks: list[Block], per_block_timings: list[list[dict]],
                  audio_filename: str, output_path: Path,
                  refs: dict[int, str] | None = None,
                  figures: dict[int, str] | None = None):
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
<html lang="en">
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
          line-height:0;white-space:nowrap}}
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
    <source src="{audio_filename}" type="{audio_type}">
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

/* ── Chapter markers ── */
player.addEventListener('loadedmetadata',()=>{{
  const dur=player.duration;
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
}});

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
                    help="Kokoro voice (default: af_heart)")
    ap.add_argument("--whisper-model", default="base",
                    help="Whisper model size: tiny/base/small/medium (default: base)")
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="Limit to first N blocks (for quick tests)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_dir) if args.output_dir else pdf_path.parent / (pdf_path.stem + "_tts")
    voice = args.voice
    whisper_model = args.whisper_model

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "audio.wav"
    html_path = output_dir / "viewer.html"

    # --- Parse PDF ---
    print(f"\nParsing {pdf_path.name}…")
    blocks = parse_pdf(str(pdf_path), max_blocks=args.max_blocks)
    if not blocks:
        print("ERROR: no readable text found in PDF.")
        sys.exit(1)

    total_words = sum(len(b.tts_text.split()) for b in blocks)
    est_min = total_words / 150
    print(f"  {len(blocks)} blocks, ~{total_words} words, ~{est_min:.1f} min audio")

    # --- TTS ---
    if wav_path.exists():
        print(f"\nAudio already exists ({wav_path}), skipping TTS.")
        full_audio, _ = sf.read(str(wav_path), dtype="float32")
        total_samples = len(full_audio)
        block_offsets = _estimate_offsets(blocks, total_samples)
    else:
        print(f"\nRunning TTS (voice={voice})…")
        full_audio, block_offsets = run_tts(blocks, voice=voice)
        print(f"  Writing {wav_path}…")
        sf.write(str(wav_path), full_audio, SAMPLE_RATE)

    # --- Whisper ---
    timestamps_path = output_dir / "timestamps.json"
    if timestamps_path.exists():
        print(f"\nLoading cached timestamps from {timestamps_path}…")
        whisper_words = json.loads(timestamps_path.read_text())
    else:
        print(f"\nRunning Whisper (model={whisper_model})…")
        whisper_words = get_word_timestamps(full_audio, model_size=whisper_model)
        timestamps_path.write_text(json.dumps(whisper_words, indent=2))
        print(f"  {len(whisper_words)} word timestamps saved.")

    # --- Assign timings to display words ---
    per_block_timings = assign_timings(blocks, whisper_words, block_offsets)

    # --- Extract references and figures (fast, no reprocessing needed) ---
    print(f"\nExtracting references and figures from PDF…")
    refs = extract_references(str(pdf_path))
    figures = extract_figure_images(str(pdf_path))
    print(f"  {len(refs)} references, {len(figures)} figures extracted")

    # --- Generate HTML ---
    print(f"Generating HTML viewer…")
    generate_html(blocks, per_block_timings, wav_path.name, html_path,
                  refs=refs, figures=figures)

    # --- Summary ---
    dur = len(full_audio) / SAMPLE_RATE
    print(f"\n{'─'*50}")
    print(f"  Audio : {wav_path}  ({dur/60:.1f} min)")
    print(f"  Viewer: {html_path}")
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
