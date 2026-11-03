# SPDX-License-Identifier: MIT
"""CPU protocol fixture, not a model engine. Exercises load context, cancellation and cleanup."""
import os
from pathlib import Path
import signal
import sys
import time
from arsenal_worker import Worker, file_result

kind, scenario, marker = sys.argv[1:4]
Path(marker).write_text(str(os.getpid()))
if scenario.startswith("hang"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if scenario == "slow-hello":
    time.sleep(0.3)
w = Worker(kind, runner="native")
config = {}

@w.op("load")
def load(request):
    global config
    if scenario == "hang-load":
        time.sleep(60)
    config = request.get("config", {})
    return {"context": config.get("context_limit", 8192), "engine": "protocol-fixture"}

@w.op("inspect_load")
def inspect(request):
    return {"config": config, "pid": os.getpid()}

@w.op("chat")
def chat(request):
    if scenario == "hang-run":
        time.sleep(60)
    return {"text": "protocol fixture"}

@w.op("convert")
def convert(request):
    if scenario == "hang-convert":
        time.sleep(60)
    path = Path(request["dst_dir"]) / "weights.bin"
    path.write_bytes(b"\1\2\3")
    return {"files": [file_result(path)]}

w.run()
