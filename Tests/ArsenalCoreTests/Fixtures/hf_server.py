# SPDX-License-Identifier: MIT
"""Loopback-only stand-in for resumable Hub and commit-addressed catalog files."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
# A blocked stdin read consumes no CPU and ends the fixture if its test host crashes.
import os as _os, sys as _sys, threading as _threading
_threading.Thread(target=lambda: (_sys.stdin.buffer.read(), _os._exit(0)), daemon=True).start()

from pathlib import Path
import sys, time, socket
ROOT = Path(sys.argv[1])
PAYLOAD = b"arsenal-test-weights\n" * 8192
COUNTS = {}
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def refuse(self, status, code=None):
        self.send_response(status)
        if code: self.send_header('X-Error-Code', code)
        self.send_header('Content-Length', '0'); self.end_headers()
    def gate(self, mode):
        """Gated repositories: `gated` refuses always (401 without a token, 403 GatedRepo with one); `gated-after-N`
        serves after N refusals; `anon401` is a bare 401; `forbidden` a 403 without a code; `missing` a 404;
        `not-found` a 401 RepoNotFound; `bad-token` a bare 401 to any token (an expired one) and the file without. Records only whether a token came, never its value. True when it answered."""
        token = bool(self.headers.get('Authorization'))
        if mode.startswith('gated') and mode != 'gated-bad-token':
            (ROOT / f'auth-{mode}').write_text('token' if token else 'none')
            limit = int(mode.split('-')[-1]) if mode.startswith('gated-after-') else None
            if limit is None or COUNTS[mode] <= limit:
                self.refuse(403 if token else 401, 'GatedRepo'); return True
            return False
        if mode == 'anon401': self.refuse(401); return True
        if mode == 'bad-token' and token: self.refuse(401); return True
        if mode == 'gated-bad-token': self.refuse(401) if token else self.refuse(401, 'GatedRepo'); return True
        if mode == 'forbidden': self.refuse(403); return True
        if mode == 'missing': self.refuse(404); return True
        if mode == 'not-found': self.refuse(401, 'RepoNotFound'); return True
        return False
    def do_GET(self):
        pieces = self.path.split('/')
        if self.path.startswith('/api/') and '/gated-meta' in self.path:
            self.refuse(403 if self.headers.get('Authorization') else 401, 'GatedRepo'); return
        if self.path.startswith('/api/') and '/bad-token' in self.path and self.headers.get('Authorization'):
            self.refuse(401); return
        if self.path.startswith('/api/') and '/redirect-out' in self.path:
            # Another host (localhost, not 127.0.0.1): a token must not follow.
            self.send_response(302); self.send_header('Location', f'http://localhost:{self.server.server_port}/echo-auth')
            self.send_header('Content-Length', '0'); self.end_headers(); return
        if self.path.startswith('/echo-auth'):
            (ROOT / 'auth-redirect').write_text('token' if self.headers.get('Authorization') else 'none')
            data = b'{"gated":"auto"}'
            self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data); return
        if '/resolve/' not in self.path:
            # HTTPSCatalogSource test routes include commit/models/...; enforce filesystem containment.
            path = ROOT.joinpath(*pieces[2:])
            if not path.is_file() or ROOT.resolve() not in path.resolve().parents:
                self.send_error(404); return
            data = path.read_bytes(); self.send_response(200); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data); return
        mode = pieces[2]
        COUNTS[mode] = COUNTS.get(mode,0)+1
        if self.gate(mode): return
        if mode == 'denied': self.send_error(401); return
        if mode == 'wrong' or (mode == 'wrong-once' and COUNTS[mode] == 1): data = PAYLOAD[:-1]+b'x'
        else: data = PAYLOAD
        if self.path.endswith('/config.json'): data = b"arsenal-config\n" * 32
        if mode == 'required-range' and not self.headers.get('Range'):
            self.send_error(400); return
        offset = int(self.headers.get('Range','bytes=0-')[6:].split('-')[0])
        if mode == 'restart': offset = 0
        self.send_response(206 if offset else 200)
        self.send_header('Content-Length',str(len(data)-offset)); self.send_header('ETag','"fixed"')
        if offset: self.send_header('Content-Range',f'bytes {offset + (1 if mode == "bad-range" else 0)}-{len(data)-1}/{len(data)}')
        self.end_headers()
        if mode == 'stalled': time.sleep(.4); return
        if mode == 'interrupted' and COUNTS[mode] == 1:
            self.wfile.write(data[:len(data)//2]); self.wfile.flush(); self.connection.shutdown(socket.SHUT_RDWR); self.connection.close(); return
        try:
            for i in range(offset,len(data),1024):
                self.wfile.write(data[i:i+1024]); self.wfile.flush()
                if mode == 'slow': time.sleep(.015)
        except (BrokenPipeError, ConnectionResetError): pass
server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
print(server.server_port,flush=True)
server.serve_forever()
