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

from parser import Block, parse_pdf

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
        for _, _, audio in pipeline(block.text, voice=voice, speed=speed):
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

        # Display words
        display_words = block.text.split()
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

def generate_html(blocks: list[Block], per_block_timings: list[list[dict]],
                  audio_filename: str, output_path: Path):
    # Flatten all (start, end) pairs into one JS array indexed by global word index
    all_timings: list[dict] = []
    for timings in per_block_timings:
        all_timings.extend(timings)

    timings_json = json.dumps(
        [{"s": round(t["start"], 3), "e": round(t["end"], 3)} for t in all_timings]
    )

    audio_type = "audio/wav" if audio_filename.endswith(".wav") else "audio/mpeg"

    # Build content HTML: each word gets a <span class="w" data-i="N">
    word_idx = 0
    content_parts: list[str] = []

    for block, timings in zip(blocks, per_block_timings):
        tokens = re.findall(r"\S+|\s+", block.text)
        inner_parts: list[str] = []
        for token in tokens:
            if token.strip():
                inner_parts.append(f'<span class="w" data-i="{word_idx}">{_esc(token)}</span>')
                word_idx += 1
            else:
                inner_parts.append(token)
        inner = "".join(inner_parts)

        if block.type == "title":
            content_parts.append(f"<h1>{inner}</h1>")
        elif block.type == "heading":
            content_parts.append(f"<h2>{inner}</h2>")
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
         color:#111;background:#faf9f7}}
    #bar{{position:sticky;top:0;z-index:100;background:#fff;
          border-bottom:1px solid #ddd;padding:8px 12px;display:flex;
          flex-direction:column;gap:4px;box-shadow:0 1px 4px rgba(0,0,0,.07)}}
    audio{{width:100%;height:44px}}
    #info{{font-family:-apple-system,sans-serif;font-size:11px;color:#888;
           display:flex;justify-content:space-between;padding:0 2px}}
    #content{{max-width:700px;margin:0 auto;padding:20px 16px 100px}}
    h1{{font-size:1.45em;margin:0.6em 0 0.5em;line-height:1.3}}
    h2{{font-size:1.15em;color:#2c2c2c;margin:1.6em 0 0.35em;
        border-bottom:1px solid #e8e8e8;padding-bottom:3px}}
    p{{margin-bottom:1.1em}}
    .w{{border-radius:3px;padding:0 1px;cursor:pointer;
        transition:background 0.07s,color 0.07s}}
    .w:hover{{background:#f0e8aa}}
    .w.active{{background:#ffe566;color:#000}}
    .w.past{{color:#bbb}}
    @media(prefers-color-scheme:dark){{
      body{{background:#1c1c1e;color:#e5e5ea}}
      #bar{{background:#1c1c1e;border-color:#3a3a3c;box-shadow:none}}
      h2{{color:#c7c7cc;border-color:#3a3a3c}}
      .w:hover{{background:#3a3a3c}}
      .w.active{{background:#b8860b;color:#fff}}
      .w.past{{color:#555}}
    }}
  </style>
</head>
<body>
<div id="bar">
  <audio id="player" controls preload="auto">
    <source src="{audio_filename}" type="{audio_type}">
  </audio>
  <div id="info">
    <span id="wpos">Word 0 / {total_words}</span>
    <span id="tpos">0:00 / 0:00</span>
  </div>
</div>
<div id="content">
{content_html}
</div>
<script>
const T={timings_json};
const player=document.getElementById('player');
const wposEl=document.getElementById('wpos');
const tposEl=document.getElementById('tpos');
const spans=document.querySelectorAll('.w');
let cur=-1,scrollLock=false;

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
  return Math.floor(s/60)+':'+(''+(Math.floor(s)%60)).padStart(2,'0');
}}

player.addEventListener('timeupdate',()=>{{
  const t=player.currentTime;
  tposEl.textContent=fmt(t)+' / '+fmt(player.duration);
  const idx=bs(t);
  if(idx===cur)return;
  if(cur>=0&&cur<spans.length){{spans[cur].classList.remove('active');spans[cur].classList.add('past');}}
  cur=idx;
  if(idx>=0&&idx<spans.length){{
    spans[idx].classList.add('active');
    if(!scrollLock)spans[idx].scrollIntoView({{behavior:'smooth',block:'center'}});
    wposEl.textContent='Word '+(idx+1)+' / {total_words}';
  }}
}});

// Reset past-word shading on seek
player.addEventListener('seeked',()=>{{
  const t=player.currentTime;
  spans.forEach((s,i)=>{{
    s.classList.remove('active','past');
    if(i<T.length&&T[i].e<t)s.classList.add('past');
  }});
  cur=-1;
}});

// Tap word → seek to it
spans.forEach((span,i)=>{{
  span.addEventListener('click',()=>{{
    if(i<T.length){{player.currentTime=T[i].s;player.play();}}
  }});
}});

// Pause auto-scroll when user scrolls manually, resume after 3 s
let scrollTimer;
window.addEventListener('scroll',()=>{{
  scrollLock=true;
  clearTimeout(scrollTimer);
  scrollTimer=setTimeout(()=>{{scrollLock=false;}},3000);
}},{{passive:true}});
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

    total_words = sum(len(b.text.split()) for b in blocks)
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

    # --- Generate HTML ---
    print(f"\nGenerating HTML viewer…")
    generate_html(blocks, per_block_timings, wav_path.name, html_path)

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
    word_counts = [max(len(b.text.split()), 1) for b in blocks]
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
