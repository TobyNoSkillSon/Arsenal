# SPDX-License-Identifier: MIT
import json
import os
import struct
import sys
from arsenal_worker import read_frame, write_frame
write_frame({"event":"hello","proto":1,"worker":"decisions","runner":"native","pid":os.getpid(),"build":"fixture","kinds":["decisions"],"ops":["decide","unload"]}, sys.stdout.buffer)
while (r := read_frame(sys.stdin.buffer)) is not None:
    i = r["id"]
    if r["op"] == "unload":
        write_frame({"id":i,"ok":True},sys.stdout.buffer)
        break
    if r.get("body") == "error":
        payload = ('{"id":%d,"ok":false,"error":{"code":"wrong_model","message":"Use another model.","model":"laya-multilingual","status":409,"body":{"é":1,"e\\u0301":2}}}' % i).encode()
    else:
        payload = ('{"answers":{"é":1,"e\\u0301":2},"id":%d,"ok":true,"usage":{"x":0.1234567890123456789}}' % i).encode()
    body = b"\x01" + payload
    sys.stdout.buffer.write(struct.pack(">I",len(body))+body)
    sys.stdout.buffer.flush()
