# SPDX-License-Identifier: MIT
import concurrent.futures
import ctypes
import hashlib
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback
from .wire import BinaryChunk, FrameError, read_frame, write_frame

class WorkerError(Exception):
    def __init__(self, code, message, *, status=None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = " ".join(str(message).splitlines())

def memory(mlx_active_mb=0, mlx_cache_mb=0):
    footprint = None
    if sys.platform == "darwin":
        try:
            lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            lib.mach_task_self.restype = ctypes.c_uint
            words = (ctypes.c_uint32 * 128)()
            count = ctypes.c_uint32(128)
            if lib.task_info(lib.mach_task_self(), 22, ctypes.byref(words), ctypes.byref(count)) == 0:
                footprint = ctypes.cast(words, ctypes.POINTER(ctypes.c_uint64))[18] / 1e6
        except (OSError, AttributeError):
            pass
    elif Path("/proc/self/statm").exists():
        footprint = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    if footprint is None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        footprint = peak / (1e6 if sys.platform == "darwin" else 1000)
    return {"footprint_mb": footprint, "mlx_active_mb": mlx_active_mb, "mlx_cache_mb": mlx_cache_mb}

def job_dir(request=None, job_id=None):
    root = Path(os.environ.get("ARSENAL_SUPPORT_DIR", str(Path.home() / "Library/Application Support/Arsenal"))) / "jobs"
    if request and request.get("out_dir"):
        path = Path(request["out_dir"])
        if not path.is_absolute():
            raise WorkerError("bad_path", "out_dir must be absolute.")
        # The supervisor creates job dirs; reject symlink escape from that root.
        if root.resolve() not in path.resolve().parents:
            raise WorkerError("bad_path", "out_dir must be inside Arsenal jobs.")
    else:
        identifier = str(job_id or (request or {}).get("id", "job"))
        if not identifier or identifier in (".", "..") or "/" in identifier or "\\" in identifier:
            raise WorkerError("bad_path", "Invalid job id.")
        path = root / identifier
    path.mkdir(parents=True, exist_ok=True)
    return path

def file_result(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise WorkerError("bad_path", "Output is not a regular file.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}

class Worker:
    def __init__(self, worker, runner="python", kinds=None, build="dev", source=None, destination=None):
        self.worker = worker
        self.runner = runner
        self.kinds = kinds or [worker]
        self.build = build
        self.source = source or sys.stdin.buffer
        self.destination = destination or sys.stdout.buffer
        self.handlers = {}
        self.jobs = {}
        self.accepted = {}
        self.lock = threading.Lock()
        self.op("ping")(lambda request: {})
        self.op("status")(lambda request: {"memory": memory()})
        self.op("unload")(lambda request: {})

    def op(self, name):
        def register(handler):
            self.handlers[name] = handler
            return handler
        return register
    on = op

    def send(self, value):
        write_frame(value, self.destination)

    def emit(self, request, event, **fields):
        if event == "progress" and "phase" not in fields:
            fields["phase"] = "step"
        if event == "progress":
            for key in ("label", "unit"):
                if key in fields and (not isinstance(fields[key], str) or len(fields[key]) > 128):
                    raise WorkerError("invalid_progress", f"{key} must be a string up to 128 characters")
        self.send({**fields, "id": request["id"], "event": event})

    def binary(self, stream, data):
        self.send(BinaryChunk(stream, bytes(data)))

    def status(self, **values):
        self.send({"event": "status", "memory": memory(**values)})

    def cancelled(self, request):
        with self.lock:
            event = self.jobs.get(request["id"])
        return event is None or event.is_set()

    def check_cancelled(self, request):
        if self.cancelled(request):
            raise WorkerError("cancelled", "Request cancelled.")

    def _dispatch(self, request):
        identifier, operation = request["id"], request["op"]
        scheduled = time.monotonic()
        instrumented = operation not in ("ping", "status", "unload")
        with self.lock:
            accepted = self.accepted.get(identifier, scheduled)
        mx = sys.modules.get("mlx.core")
        if instrumented:
            try:
                if mx is not None and hasattr(mx, "reset_peak_memory"):
                    mx.reset_peak_memory()
                else:
                    mx = None
            except Exception:
                mx = None  # Instrumentation failure must not change handler behavior.
            self.emit(request, "progress", phase="loading" if operation == "load" else "prefill" if operation == "chat" else "step")
        try:
            handler = self.handlers.get(operation)
            if handler is None:
                raise WorkerError("unsupported_op", "Unsupported operation.")
            result = handler(request) or {}
            if not isinstance(result, dict):
                raise WorkerError("invalid_reply", "Handler must return an object.")
            self.check_cancelled(request)
            if instrumented:
                timing = {"queue_s": max(0, scheduled - accepted)}
                if operation == "load":
                    timing["load_s"] = time.monotonic() - scheduled
                if mx is not None and hasattr(mx, "get_peak_memory"):
                    try:
                        timing["peak_mb"] = mx.get_peak_memory() / 1e6
                    except Exception:
                        pass
                timing.update(result.get("timing", {}))
                result["timing"] = timing
                self.emit(request, "progress", phase="done", timing=timing)
            self.status()
            self.send({**result, "id": identifier, "ok": True})
        except WorkerError as error:
            self.send({"id": identifier, "ok": False, "error": {"code": error.code, "message": error.message,
                      **({"status": error.status} if error.status is not None else {})}})
        except Exception:
            traceback.print_exc(file=sys.stderr)
            self.send({"id": identifier, "ok": False, "error": {"code": "runner", "message": "Runner request failed; see its log."}})
        finally:
            with self.lock:
                self.jobs.pop(identifier, None)
                self.accepted.pop(identifier, None)

    def run(self):
        self.send({"event": "hello", "proto": 1, "worker": self.worker, "runner": self.runner, "pid": os.getpid(), "build": self.build, "kinds": self.kinds, "ops": sorted(set(self.handlers) | {"cancel"})})
        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="arsenal") as executor:
            while True:
                request = read_frame(self.source)
                if request is None:
                    break
                if not isinstance(request, dict) or type(request.get("id")) is not int or request["id"] < 1 or not isinstance(request.get("op"), str):
                    raise FrameError("Invalid request.")
                if request["op"] == "cancel":
                    with self.lock:
                        event = self.jobs.get(request.get("target"))
                        if event:
                            event.set()
                    self.send({"id": request["id"], "ok": True})
                    continue
                with self.lock:
                    if request["id"] in self.jobs:
                        raise FrameError("Duplicate request id.")
                    if len(self.jobs) >= 64:
                        self.send({"id": request["id"], "ok": False, "error": {"code": "busy", "message": "Runner queue is full."}})
                        continue
                    self.jobs[request["id"]] = threading.Event()
                    self.accepted[request["id"]] = time.monotonic()
                if request["op"] not in ("ping", "status", "unload"):
                    self.emit(request, "progress", phase="queued")
                future = executor.submit(self._dispatch, request)
                if request["op"] == "unload":
                    future.result()
                    break
        return 0
    serve = run
