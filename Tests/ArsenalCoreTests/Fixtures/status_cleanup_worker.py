# CPU-only fixture: holds Supervisor's detach/service-resident gap until the test releases it.
import json, os, pathlib, struct, sys, time

gate = pathlib.Path(sys.argv[1])
def send(value):
    body = b'\1' + json.dumps(value).encode()
    sys.stdout.buffer.write(struct.pack('>I', len(body)) + body)
    sys.stdout.buffer.flush()

ops = ['load', 'status', 'cancel', 'unload', 'ping', 'chat', 'count_tokens']
send(dict(event='hello', proto=1, worker='stub', runner='native', pid=os.getpid(), build='test', kinds=['chat'], ops=ops))
while header := sys.stdin.buffer.read(4):
    body = sys.stdin.buffer.read(struct.unpack('>I', header)[0])
    request = json.loads(body[1:])
    if request['op'] == 'unload':
        (gate / 'entered').touch()
        deadline = time.monotonic() + 3
        while not (gate / 'release').exists() and time.monotonic() < deadline:
            time.sleep(0.002)
    send(dict(id=request['id'], ok=True, memory=dict(footprint_mb=1, mlx_active_mb=0, mlx_cache_mb=0)))
    if request['op'] == 'unload':
        break
