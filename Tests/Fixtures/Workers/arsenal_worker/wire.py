# SPDX-License-Identifier: MIT
import dataclasses
import json
import struct
import threading

MAX_FRAME = 16 * 1024 * 1024
_WRITE_LOCK = threading.Lock()

class FrameError(ValueError):
    pass

@dataclasses.dataclass(frozen=True)
class BinaryChunk:
    stream: int
    data: bytes

def _exact(source, count, eof=False):
    chunks = bytearray()
    while len(chunks) < count:
        data = source.read(count - len(chunks))
        if not data:
            if eof and not chunks:
                return None
            raise FrameError("Truncated frame.")
        chunks.extend(data)
    return bytes(chunks)

def read_frame(source):
    header = _exact(source, 4, eof=True)
    if header is None:
        return None
    size = struct.unpack(">I", header)[0]
    if not 0 < size <= MAX_FRAME:
        raise FrameError("Invalid frame length.")
    body = _exact(source, size)
    if body[0] == 1:
        try:
            value = json.loads(body[1:].decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(FrameError("Nonfinite JSON number.")))
        except (ValueError, UnicodeError) as error:
            raise FrameError("Malformed JSON frame.") from error
        if not isinstance(value, dict):
            raise FrameError("JSON frame must contain an object.")
        return value
    if body[0] == 2 and size >= 5:
        return BinaryChunk(struct.unpack(">I", body[1:5])[0], body[5:])
    raise FrameError("Unknown or malformed frame kind.")

def write_frame(value, destination):
    if isinstance(value, BinaryChunk):
        body = b"\x02" + struct.pack(">I", value.stream) + value.data
    elif isinstance(value, dict):
        body = b"\x01" + json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    else:
        raise FrameError("Expected object or BinaryChunk.")
    if len(body) > MAX_FRAME:
        raise FrameError("Frame exceeds 16 MiB.")
    data = struct.pack(">I", len(body)) + body
    with _WRITE_LOCK:
        offset = 0
        while offset < len(data):
            count = destination.write(data[offset:])
            if count is None:  # BufferedIOBase accepted the full write.
                offset = len(data)
            elif count > 0:
                offset += count
            else:
                raise FrameError("Frame output stopped accepting bytes.")
        destination.flush()
