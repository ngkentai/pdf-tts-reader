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
cd pdf_tts && python3 serve.py 8080
# Then scan the QR code or open http://<local-ip>:8080/viewer.html
```

The viewer works in any mobile browser. Features:
- Sticky audio player at top
- Highlighted current word (yellow)
- Grayed-out already-read words
- Tap any word to seek to it
- Auto-scroll pauses for 3 s when you manually scroll
- Dark mode support

## iOS app (offline reading)

`ios/PDFTTSReader.xcodeproj` is a SwiftUI app that downloads documents to the
phone so nothing needs to stream. It reuses each document's `viewer.html`
unchanged — same word highlighting, tap-to-seek, and chapter marks.

1. On the computer, serve this folder: `cd pdf_tts && python3 serve.py 8080`
   (not `python3 -m http.server` — it lacks HTTP Range support, which breaks
   audio seeking for browser streaming)
2. Open the project in Xcode, set your signing team, build to your iPhone.
3. In the app, set the server URL (defaults to `http://192.168.1.121:8080`),
   pull to refresh, and tap ⬇ next to a document. It downloads `viewer.html`
   plus the MP3 (sizes come from `manifest.json`) into the app's storage.
4. Read/listen fully offline. Playback position is saved natively (shared
   with the viewer's own resume logic) and audio keeps playing with the
   screen locked. Swipe a downloaded document to delete it.

Older manifests without the `audio`/`*_bytes` fields still work — the app
falls back to extracting the audio filename from the downloaded viewer.

## Known limitations (prototype)

- **Multi-column PDFs**: PyMuPDF may interleave columns; use single-column or converted PDFs for best results.
- **Word timing accuracy**: Whisper words are mapped to display words by position within each block. Expansions like "2023" → "twenty twenty three" may shift highlighting by 1–2 words locally.
- **Math / equations**: skipped (non-text blocks). Formula text may be garbled.
- **Scanned PDFs**: use an OCR step first (e.g. `ocrmypdf input.pdf output.pdf`).
