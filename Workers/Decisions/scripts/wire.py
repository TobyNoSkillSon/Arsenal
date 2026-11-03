# SPDX-License-Identifier: MIT
"""Minimal ArsenalWire v1 client for the decisions worker (tests and parity runs). Frames: u32 BE length, u8 kind, JSON.

Replies are decoded with object_pairs_hook so key order and byte-distinct keys survive (as the worker sends them).
"""
import json
import os
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get('ARSENAL_DECISIONS_WORKER', ROOT / '.build/arm64-apple-macosx/release/arsenal-worker-decisions'))
WEIGHTS = Path(os.environ.get('ARSENAL_DECISIONS_WEIGHTS', Path.home() / 'Library/Application Support/Arsenal-dev/weights/decisions'))
MODELS = ROOT.parents[1] / 'models'


def pairs(items):
    return dict(items)


class Worker:
    def __init__(self, support='/tmp/arsenal-decisions/support', env=None):
        Path(support).mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, ARSENAL_SUPPORT_DIR=support, ARSENAL_CACHE_DIR=support + '/cache', **(env or {}))
        environment.pop('MLX_ENABLE_TF32', None)
        self.stderr = open(Path(support) / 'worker.stderr.log', 'a')
        self.proc = subprocess.Popen([str(BINARY)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, env=environment)
        self.next = 0
        self.hello = self.read()
        self.events = []

    def read(self):
        head = self.proc.stdout.read(5)
        if len(head) < 5:
            raise EOFError('worker closed its stream')
        length, kind = struct.unpack('>IB', head)
        payload = self.proc.stdout.read(length - 1)
        assert kind == 1, kind
        return json.loads(payload, object_pairs_hook=pairs)

    def send_raw(self, payload: bytes):
        self.proc.stdin.write(struct.pack('>IB', len(payload) + 1, 1) + payload)
        self.proc.stdin.flush()

    def call(self, op, **params):
        self.next += 1
        rid = self.next
        self.send_raw(json.dumps(dict(params, id=rid, op=op), ensure_ascii=False).encode())
        while True:
            frame = self.read()
            if frame.get('id') == rid:
                return frame
            self.events.append(frame)

    def load(self, model, dtype='fp16', fast=True, weights=None, config=None):
        weights = weights or WEIGHTS / model / dtype
        start = time.time()
        reply = self.call('load', model_dir=str(MODELS / model), weights_dir=str(weights), dtype=dtype, fast=fast, config=config or {})
        reply['wall_s'] = round(time.time() - start, 2)
        return reply

    def close(self):
        if self.proc.poll() is None:
            try:
                self.call('unload')
            except (EOFError, BrokenPipeError):
                pass
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait()
        self.stderr.close()
        return self.proc.returncode
