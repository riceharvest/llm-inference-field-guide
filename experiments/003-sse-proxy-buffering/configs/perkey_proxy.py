#!/usr/bin/env python3
"""SSE proxy for llama-server with small read buffer for real-time streaming.

The bug: Python's HTTPResponse.read(65536) waits to fill a 64 KiB buffer
before yielding tokens. This hides real-time generation behind a
fill-then-release pattern — the user sees a long pause, then the entire
reasoning block appears at once.

Fix: Use a small read buffer (128 bytes) for real-time streaming.
"""

import http.server
import urllib.request
import urllib.error
import json
import sys
import os

# Configuration
UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8080")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8081"))
CHUNK = 128  # small buffer = real-time streaming (was 65536)

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.proxy_request("GET")

    def do_POST(self):
        self.proxy_request("POST")

    def proxy_request(self, method):
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Build upstream request
        url = f"{UPSTREAM}{self.path}"
        req = urllib.request.Request(url, data=body, method=method)
        for key, value in self.headers.items():
            if key.lower() not in ("host", "transfer-encoding"):
                req.add_header(key, value)

        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                # Send response status
                self.send_response(response.status)

                # Copy response headers
                for key, value in response.headers.items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)

                # Ensure streaming content type is preserved
                if "text/event-stream" in response.headers.get("Content-Type", ""):
                    self.send_header("Content-Type", "text/event-stream")

                self.end_headers()

                # Stream with small buffer for real-time output
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"Proxy error: {e}".encode())

    def log_message(self, format, *args):
        # Suppress default logging
        pass

if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", LISTEN_PORT), ProxyHandler)
    print(f"Proxy listening on :{LISTEN_PORT} → {UPSTREAM}")
    print(f"Read buffer: {CHUNK} bytes")
    server.serve_forever()
