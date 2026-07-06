#!/usr/bin/env python3
"""Static file server with HTTP Range support.

`python3 -m http.server` ignores Range headers, so browsers cannot seek
inside streamed MP3s (clicking a word or the progress bar snaps back to
the current position). This drop-in replacement serves the pdf_tts
folder with byte-range support.

Usage: python3 serve.py [port]     (default 8080)
"""
import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):

    def send_head(self):
        self._range_remaining = None
        path = self.translate_path(self.path)
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if os.path.isdir(path) or not os.path.isfile(path) or not m \
                or (not m.group(1) and not m.group(2)):
            self._advertise_ranges = os.path.isfile(path)
            return super().send_head()

        size = os.path.getsize(path)
        if m.group(1):
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        else:  # suffix range: last N bytes
            start = max(0, size - int(m.group(2)))
            end = size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._range_remaining = end - start + 1
        return f

    def end_headers(self):
        if getattr(self, "_advertise_ranges", False):
            self.send_header("Accept-Ranges", "bytes")
            self._advertise_ranges = False
        super().end_headers()

    def copyfile(self, source, outputfile):
        remaining = self._range_remaining
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    root = os.path.dirname(os.path.abspath(__file__))
    handler = partial(RangeHandler, directory=root)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"Serving {root} on port {port} (with Range support)")
    server.serve_forever()


if __name__ == "__main__":
    main()
