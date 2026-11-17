#!/usr/bin/env python3
"""
dev_server.py
=============
SANADA Unified Development Server & Reverse Proxy
Serves the frontend static files and seamlessly proxies /api and /static/stems
requests to the FastAPI backend (http://127.0.0.1:8000).

Allows Cloudflare Tunnel (cloudflared) to forward both frontend and backend
on a single public URL.
"""

import os
import sys
import http.server
import requests

PORT = int(os.environ.get("PORT", 3000))
BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend"))

class UnifiedProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_proxy(self):
        target_url = f"{BACKEND}{self.path}"
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else None

        try:
            resp = requests.request(
                method=self.command,
                url=target_url,
                headers=headers,
                data=body,
                stream=True,
                allow_redirects=False,
                timeout=120
            )
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() not in ("content-length", "transfer-encoding", "content-encoding", "connection"):
                    self.send_header(k, v)
            resp_bytes = resp.content
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(f'{{"error": "Backend offline or unreachable at {BACKEND}", "details": "{str(e)}"}}\n'.encode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/api/") or self.path.startswith("/static/stems") or self.path.startswith("/static/demos"):
            self.do_proxy()
        else:
            super().do_GET()

    def do_HEAD(self):
        if self.path.startswith("/api/") or self.path.startswith("/static/stems") or self.path.startswith("/static/demos"):
            self.do_proxy()
        else:
            super().do_HEAD()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self.do_proxy()
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        if self.path.startswith("/api/"):
            self.do_proxy()
        else:
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

def main():
    server_address = ("0.0.0.0", PORT)
    with http.server.ThreadingHTTPServer(server_address, UnifiedProxyHandler) as httpd:
        print(f"[*] SANADA Unified Server running on http://0.0.0.0:{PORT}")
        print(f"[*] Serving frontend from: {FRONTEND_DIR}")
        print(f"[*] Proxying API requests to: {BACKEND}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    main()
