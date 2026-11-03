#!/usr/bin/env python3
"""Stream SwiftPM output unchanged; require executed tests on every execution path."""
import re
import subprocess
import sys
import tempfile


def main():
    args = sys.argv[1:]
    inspect = bool(args and args[0] == '--inspect')
    if inspect: args.pop(0)
    if len(args) < 2 or args[0] != '--': sys.exit('usage: test-run-checked.py [--inspect] -- command...')
    command = args[1:]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while chunk := process.stdout.read1(65536):
            sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush(); log.write(chunk)
        status = process.wait()
        if status: sys.exit(status if status > 0 else 128 - status)
        if inspect: return
        log.seek(0); output = log.read().decode(errors='replace')
        counts = re.findall(r'(?:Test run with|Executed) (\d+) tests?\b', output)
        if not any(int(count) > 0 for count in counts):
            sys.exit('Tests: failed or zero tests matched')


if __name__ == '__main__': main()
