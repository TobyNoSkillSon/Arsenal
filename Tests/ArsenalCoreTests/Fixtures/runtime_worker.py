# CPU-only adversarial worker, framed protocol matching the Swift stub.
import json, os, signal, struct, sys, time

def send(value):
    body = b'\1' + json.dumps(value).encode()
    sys.stdout.buffer.write(struct.pack('>I', len(body)) + body)
    sys.stdout.buffer.flush()

def read():
    header = sys.stdin.buffer.read(4)
    if not header:
        return None
    body = sys.stdin.buffer.read(struct.unpack('>I', header)[0])
    return json.loads(body[1:])

send(dict(event='hello', proto=1, worker='stub', runner='native', pid=os.getpid(), build='test', kinds=['chat'], ops=['ping', 'cancel', 'unload']))
while (r := read()) is not None:
    i, op = r['id'], r['op']
    if op == 'ack_ignore':
        continue
    if op == 'stream':
        for _ in range(r.get('chunks', 10)):
            send(dict(id=i, event='token', text='x'))
            time.sleep(0.04)
    if op == 'binary':
        send(dict(id=i, event='audio_begin', stream=7))
        for _ in range(r.get('chunks', 10)):
            body = b'\2' + struct.pack('>I', 7) + b'pcm'
            sys.stdout.buffer.write(struct.pack('>I', len(body)) + body)
            sys.stdout.buffer.flush()
            time.sleep(0.04)
    if op == 'exit_child':
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(60)
            os._exit(0)
        send(dict(id=i, ok=True, child=child))
        os._exit(0)
    if op == 'env':
        send(dict(id=i, ok=True, env=dict(os.environ)))
        continue
    if op == 'logs':
        os.write(2, b'x' * (6 * 1024 * 1024))
    send(dict(id=i, ok=True))
    if op == 'unload':
        break
