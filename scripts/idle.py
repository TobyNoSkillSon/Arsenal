#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Idle cost of running processes, without powermetrics (no root).

    idle.py [--seconds 60] [--label NAME] PID [PID...]

For each PID: CPU time (user+system) and wakeups (package-idle + interrupt) consumed during the window, read from the
kernel's per-process rusage (proc_pid_rusage, RUSAGE_INFO_V4) at the start and the end; thread count and footprint at
the end; direct children (pgrep -P). One JSON object on stdout. Why rusage and not `top`: top's %CPU is a 1-sample
rate rounded to 0.1 %, and its idlew column is a rate as well; the rusage deltas are exact counts over the window.
"""
import ctypes, json, subprocess, sys, time

libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


class RUsageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
        "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
        "diskio_byteswritten", "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
        "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive",
        "billed_system_time", "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint", "instructions",
        "cycles", "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time")]


class Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


tb = Timebase(); libc.mach_timebase_info(ctypes.byref(tb))
TICK_NS = tb.numer / tb.denom  # rusage CPU times are mach absolute ticks on Apple Silicon


def rusage(pid):
    info = RUsageV4()
    if libc.proc_pid_rusage(pid, 4, ctypes.byref(info)) != 0:
        raise SystemExit(f"proc_pid_rusage({pid}) failed")
    return info


def threads(pid):
    out = subprocess.run(["ps", "-M", "-p", str(pid)], capture_output=True, text=True).stdout.splitlines()
    return max(0, len(out) - 1)


def children(pid):
    out = subprocess.run(["pgrep", "-lP", str(pid)], capture_output=True, text=True).stdout.split("\n")
    return [line for line in out if line.strip()]


def name(pid):
    return subprocess.run(["ps", "-o", "comm=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def main():
    args = sys.argv[1:]
    seconds, label = 60.0, None
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag == "--seconds": seconds = float(args.pop(0))
        elif flag == "--label": label = args.pop(0)
    pids = [int(p) for p in args]
    start = {p: rusage(p) for p in pids}; t0 = time.monotonic()
    time.sleep(seconds)
    end = {p: rusage(p) for p in pids}; wall = time.monotonic() - t0
    result = {"label": label, "window_s": round(wall, 2), "processes": []}
    for p in pids:
        a, b = start[p], end[p]
        cpu_ns = ((b.user_time + b.system_time) - (a.user_time + a.system_time)) * TICK_NS
        wakeups = (b.pkg_idle_wkups - a.pkg_idle_wkups) + (b.interrupt_wkups - a.interrupt_wkups)
        result["processes"].append({
            "pid": p, "command": name(p),
            "cpu_ms": round(cpu_ns / 1e6, 3), "cpu_percent": round(100 * cpu_ns / 1e9 / wall, 4),
            "wakeups": wakeups, "wakeups_per_s": round(wakeups / wall, 3),
            "threads": threads(p), "footprint_mb": round(b.phys_footprint / 1048576, 1),
            "disk_written_bytes": b.diskio_byteswritten - a.diskio_byteswritten,
            "children": children(p),
        })
    print(json.dumps(result))


if __name__ == "__main__":
    main()
