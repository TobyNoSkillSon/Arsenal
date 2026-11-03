# SPDX-License-Identifier: MIT
"""Arsenal v1 stdio runner support, Python >= 3.10, standard library only.

    worker = Worker(worker="images", runner="mflux", kinds=["images"])
    @worker.op("generate")
    def generate(request):
        worker.emit(request, "progress", step=1, total=1)
        return {"images": [...]}
    worker.run()

Handlers receive the complete request dictionary. Return a reply dictionary
(without id/ok), or raise WorkerError. stdout belongs exclusively to frames.
"""
from .wire import BinaryChunk, FrameError, MAX_FRAME, read_frame, write_frame
from .worker import Worker, WorkerError, memory, job_dir, file_result

__all__ = ["Worker", "WorkerError", "BinaryChunk", "FrameError", "MAX_FRAME", "read_frame", "write_frame", "memory", "job_dir", "file_result"]
