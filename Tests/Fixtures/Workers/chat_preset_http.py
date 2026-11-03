# SPDX-License-Identifier: MIT
"""CPU-only HTTP fixture for the chat tool's complete preset request path."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
root = Path(sys.argv[1])
def tokens(messages):
    return 49 + sum(len(m['content'].split()) * 2 for m in messages)
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with (root / 'requests.jsonl').open('a') as handle:
            handle.write(json.dumps({'path': self.path, 'body': body}) + '\n')
        self.send_response(200)
        if self.path.endswith('count_tokens'):
            self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps({'input_tokens': tokens(body['messages'])}).encode())
        else:
            self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            chunk = {'choices': [{'delta': {'content': 'Fixture reply'}, 'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': tokens(body['messages']), 'completion_tokens': 2}}
            self.wfile.write(('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode())
server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
port_temporary = root / 'port.tmp'
port_temporary.write_text(str(server.server_port))
port_temporary.replace(root / 'port')
server.serve_forever()
