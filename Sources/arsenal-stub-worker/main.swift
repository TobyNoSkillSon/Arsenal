// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import CryptoKit
import Darwin

let args = CommandLine.arguments
let kind = args.firstIndex(of: "--kind").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } ?? "chat"
let intakeFixture = args.contains("--intake-fixture") ? IntakeFixture(fp16Passes: args.contains("--fp16-passes")) : nil
let ops =
    [
        "load", "status", "cancel", "unload", "ping", "convert", "speak", "voices", "decide", "judge", "generate", "chat", "count_tokens", "test_config", "test_echo",
        "generate_sound",
        "generate_music", "enhance"
    ] + (intakeFixture == nil ? [] : ["measurement_source", "measurement_components"])
func send(_ value: JSONObject) throws { try FrameIO.write(.json(value), to: .standardOutput) }
func file(_ path: String, _ data: Data) throws -> JSONValue {
    let url = URL(fileURLWithPath: path); try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true); try data.write(to: url)
    return ["path": .string(path), "bytes": .number(Double(data.count)), "sha256": .string(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())]
}
try send(WorkerHello(worker: "stub", kinds: [kind], ops: ops).json)
var loaded = false
var loadedConfig: JSONObject = [:]
var loadedDtype: JSONValue = .null
var kernelReports: [JSONValue] = []
var injectedSwapBytes: JSONValue = 0
var grown: [UnsafeMutableRawPointer] = []
while let frame = try FrameIO.read(from: .standardInput) {
    guard case .json(let req) = frame, let id = req["id"], let op = req["op"]?.string else { exit(2) }
    var reply: JSONObject = ["id": id, "ok": true]
    if let result = try intakeFixture?.handle(op, request: req) {
        reply.merge(result) { _, new in new }; try send(reply)
        continue
    }
    switch op {
    case "load":
        if args.contains("--fail-load") {
            reply["ok"] = false; reply["error"] = ["code": "out_of_memory", "message": "Injected Load allocation failure."]; break
        }
        loaded = true; loadedConfig = req["config"]?.object ?? [:]; loadedDtype = req["dtype"] ?? .null
        kernelReports = (loadedConfig["kernels"]?.array ?? []).map { k in
            ["id": k["id"] ?? "stub", "replaces": k["replaces"] ?? "stub", "mode": k["mode"] ?? "exact", "state": "stock", "reason": "CPU fixture does not execute Metal."]
        }
        reply["kernels"] = .array(kernelReports); reply["load_s"] = 0.01; reply["engine"] = "stub"
    case "status", "ping", "cancel": reply["swap_used_bytes"] = injectedSwapBytes; reply["loaded"] = .bool(loaded); reply["worker_pid"] = .number(Double(getpid()))
    case "unload": loaded = false
    case "convert":
        // `--convert-grow-mb N`: a conversion that grows by N MB and then holds for 10 s (the Mac-wide memory stop's test).
        if let index = args.firstIndex(of: "--convert-grow-mb"), index + 1 < args.count, let megabytes = Int(args[index + 1]) {
            var size = 0
            while size < megabytes * 1_048_576, let block = malloc(16 * 1_048_576) { memset(block, 1, 16 * 1_048_576); grown.append(block); size += 16 * 1_048_576; usleep(10_000) }
            sleep(10)
        }
        let dst = req["dst_dir"]?.string ?? "/tmp/arsenal-stub"
        reply["files"] = .array([
            try file(dst + "/weights.bin", Data([1, 2, 3])), try file(dst + "/arsenal-snapshot.json", Data("{}".utf8)), try file(dst + "/arsenal-derived.json", Data("{}".utf8))
        ])
    case "test_memory_counters":
        injectedSwapBytes = req["swap_used_bytes"] ?? injectedSwapBytes
        reply["swap_used_bytes"] = injectedSwapBytes; reply["loaded"] = .bool(loaded); reply["worker_pid"] = .number(Double(getpid()))
    case "test_config": reply["config"] = .object(loadedConfig); reply["dtype"] = loadedDtype
    case "test_echo": reply["params"] = .object(req.filter { !["id", "op"].contains($0.key) })
    case "voices": reply["voices"] = [["id": "af_heart", "name": "Heart", "language": "en"]]
    case "speak", "generate_sound", "generate_music", "enhance":
        if req["text"]?.string == "__fixture_allocation_failure__" {
            reply["ok"] = false; reply["error"] = ["code": "out_of_memory", "message": "Injected worker allocation failure."]; break
        }
        if let out = req["out_dir"]?.string {
            reply.merge(try file(out + "/audio.pcm", Data(repeating: 0, count: 320)).object!) { _, new in new }
        } else {
            try send(["id": id, "event": "audio_begin", "stream": 1, "sample_rate": 16000, "channels": 1, "encoding": "pcm_s16le"])
            try FrameIO.write(.binary(stream: 1, bytes: Data(repeating: 0, count: 320)), to: .standardOutput)
        }
        reply["duration_s"] = 0.01; reply["gen_s"] = 0.001; reply["first_audio_s"] = 0.001; reply["rtf"] = 0.1
    case "decide":
        if req["body"] != nil { reply["_test_body"] = req["body"]; reply["_test_params"] = .object(req.filter { !["id", "op"].contains($0.key) }) }; reply["answers"] = []
        reply["model"] = "stub"; reply["usage"] = ["input_tokens": 1, "output_tokens": 1]
    case "judge": if req["body"] != nil { reply["_test_body"] = req["body"]; reply["_test_params"] = .object(req.filter { !["id", "op"].contains($0.key) }) }; reply["results"] = []
    case "generate":
        try send(["id": id, "event": "progress", "step": 1, "total": 1]); let out = req["out_dir"]?.string ?? "/tmp/arsenal-stub"
        var image = try file(out + "/image.png", Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJ1sAAAAASUVORK5CYII=")!).object!
        image["width"] = 1; image["height"] = 1; reply["images"] = [.object(image)]; reply["seed"] = req["seed"] ?? 0; reply["seconds"] = 0.01
    case "chat":
        if req["stream"]?.bool == true { try send(["id": id, "event": "token", "text": "Hello from Arsenal."]) }; reply["text"] = "Hello from Arsenal."
        reply["finish_reason"] = "stop"; reply["usage"] = ["prompt_tokens": 1, "completion_tokens": 4]; reply["ttft_s"] = 0.001; reply["tokens_per_s"] = 400
    case "count_tokens": reply["count"] = .number(Double((req["text"]?.string ?? "hello").split(separator: " ").count))
    case "test_grow":
        // A runaway for the supervisor's memory ceiling: touch `chunk_mb` more every `interval_ms` until `max_mb`.
        let chunk = Int(req["chunk_mb"]?.number ?? 32) * 1_048_576, total = Int(req["max_mb"]?.number ?? 256) * 1_048_576
        let interval = UInt32((req["interval_ms"]?.number ?? 10) * 1000)
        var size = 0
        while size < total, let block = malloc(chunk) { memset(block, 1, chunk); grown.append(block); size += chunk; usleep(interval) }
        // `hold_ms`: keep the request running at that size (a worker that stays big while the Mac fills around it).
        if let hold = req["hold_ms"]?.number, hold > 0 { usleep(UInt32(min(hold, 60_000) * 1000)) }
        reply["grown_mb"] = .number(Double(size / 1_048_576))
    case "test_live_status":
        // A runner's reading during a job (the native host sends one each second): weights, MLX and the job's MLX peak.
        try send([
            "event": "status", "live": true,
            "memory": ["footprint_mb": 900, "mlx_active_mb": 700, "mlx_cache_mb": 50, "mlx_weights_mb": 400, "mlx_peak_mb": 820]
        ])
    case "test_crash": exit(23)
    case "test_hang": sleep(60)
    case "test_ignore_term": signal(SIGTERM, SIG_IGN); sleep(60)
    case "test_garbage": try FileHandle.standardOutput.write(contentsOf: Data([0, 0, 0, 1, 99])); exit(0)
    case "test_oversize": try FileHandle.standardOutput.write(contentsOf: Data([2, 0, 0, 0])); sleep(60)
    default: reply["ok"] = false; reply["error"] = ["code": "unsupported_op", "message": "Unsupported operation."]
    }
    if ["load", "status", "unload", "speak", "generate", "chat"].contains(op) {
        let mem: JSONValue = ["footprint_mb": loaded ? 16 : 0, "footprint_peak_mb": 16, "mlx_active_mb": 0, "mlx_cache_mb": 0]
        try send(["event": "status", "memory": mem, "kernels": .array(kernelReports)]); reply["memory"] = mem
    }
    try send(reply)
    if op == "unload" { exit(0) }
}
