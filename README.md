# PDF TTS Pipeline

Converts a PDF to a synchronized audio + scrolling-text HTML viewer.

## Setup

```bash
cd pdf_tts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate
python pipeline.py paper.pdf [output_dir] [voice] [whisper_model]
```

**Defaults:** `output_dir = paper_tts/`, `voice = af_heart`, `whisper_model = base`

**Voices:**
| ID | Style |
|----|-------|
| `af_heart` | American female, warm |
| `af_sky` | American female, airy |
| `am_michael` | American male |
| `bf_emma` | British female |
| `bm_george` | British male |

**Whisper models** (larger = slower startup, more accurate timestamps):
`tiny` (39MB) → `base` (74MB) → `small` (244MB) → `medium` (769MB)

## Outputs

```
paper_tts/
├── audio.wav          # full audio file
├── timestamps.json    # cached Whisper word timestamps (re-used on re-runs)
└── viewer.html        # synchronized viewer
```

## View on phone

After the pipeline completes, it prints a command + QR code:

```bash
cd paper_tts && python3 -m http.server 8080
# Then scan the QR code or open http://<local-ip>:8080/viewer.html
```

The viewer works in any mobile browser. Features:
- Sticky audio player at top
- Highlighted current word (yellow)
- Grayed-out already-read words
- Tap any word to seek to it
- Auto-scroll pauses for 3 s when you manually scroll
- Dark mode support

## Known limitations (prototype)

- **Multi-column PDFs**: PyMuPDF may interleave columns; use single-column or converted PDFs for best results.
- **Word timing accuracy**: Whisper words are mapped to display words by position within each block. Expansions like "2023" → "twenty twenty three" may shift highlighting by 1–2 words locally.
- **Math / equations**: skipped (non-text blocks). Formula text may be garbled.
- **Scanned PDFs**: use an OCR step first (e.g. `ocrmypdf input.pdf output.pdf`).
