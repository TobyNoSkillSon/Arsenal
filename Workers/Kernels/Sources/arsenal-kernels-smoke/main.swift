// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXFast
import ArsenalKernels
import ArsenalKernelsCore

// GPU smoke for the library (run under scripts/mlock gpulock): an exact folder kernel through MLX, the Exact gate,
// a broken kernel and an oversized threadgroup falling back without killing the process, MLX still healthy after.
// Writes one line per check; exit 1 on any failure. ARSENAL_SUPPORT_DIR must point at a temp dir (verdict file).
// --bench <laya kernels dir>: dependent chains r = r + x.W^T (Laya mlp.Wo, K 2624 -> N 1024, fp16) on stock MLX, the
// splitk-exact kernel and the smallm-tile kernel; ms per GEMM (GPU + encode; one eval per chain).
if CommandLine.arguments.count == 3, CommandLine.arguments[1] == "--bench" {
    let root = URL(fileURLWithPath: CommandLine.arguments[2])
    func folder(_ id: String, _ stem: String) throws -> FolderMLXKernel {
        try FolderMLXKernel(
            try LoadedKernel.load(
                KernelSpec(
                    id: id, mode: .exact, replaces: "linear.down", entry: id.replacingOccurrences(of: "-", with: "_"),
                    files: [stem + ".h.metal", stem + ".metal"], header: stem + ".h.metal", inputs: ["x", "w", "bias", "res"], outputs: ["out"]), base: root))
    }
    let splitk = try folder("splitk-exact", "any/splitk_exact"), smallm = try folder("smallm-tile", "m5/smallm_tile")
    let k = 2624, n = 1024, steps = 64
    let w = (MLXRandom.normal([n, k], key: MLXRandom.key(3)) * 0.02).asType(.float16)
    let none = MLXArray([Float(0)])
    for m in [96, 192, 640] {
        let x = MLXRandom.normal([m, k], key: MLXRandom.key(4)).asType(.float16)
        let r0 = MLXRandom.normal([m, n], key: MLXRandom.key(5)).asType(.float16)
        eval(w, x, r0)
        func chain(_ step: (MLXArray) -> MLXArray) -> (ms: Double, out: MLXArray) {
            var r = r0; for _ in 0..<4 { r = step(r) }; eval(r) // warm-up + compile
            var best = Double.infinity
            for _ in 0..<5 {
                r = r0
                let t = Date()
                for _ in 0..<steps { r = step(r) }
                eval(r)
                best = min(best, Date().timeIntervalSince(t) * 1000 / Double(steps))
            }
            return (best, r)
        }
        let stock = chain { r in r + MLX.matmul(x, w.transposed()) }
        let partition = k <= 1024 ? k / 2 : k <= 2048 ? 1024 : k <= 4096 ? 2048 : 4096, parts = (k + partition - 1) / partition
        let fused = chain { r in
            splitk(
                [x, w, none, r], template: [("T", DType.float16), ("PS", partition), ("P", parts), ("HAS_BIAS", false), ("HAS_RES", true)],
                grid: ((m + 31) / 32 * 64 * parts, (n + 31) / 32, 1), threadGroup: (64 * parts, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [.float16])![0]
        }
        let sg = m <= 64 ? 8 : 4
        let tile: (ms: Double, out: MLXArray)? =
            m <= 256
            ? chain { r in
                smallm(
                    [x, w, none, r], template: [("T", DType.float16), ("BM", 32), ("BN", 32), ("SG", sg), ("HAS_BIAS", false), ("HAS_RES", true)],
                    grid: ((m + 31) / 32 * 32 * sg, (n + 31) / 32, 1), threadGroup: (32 * sg, 1, 1),
                    outputShapes: [[m, n]], outputDTypes: [.float16])![0]
            } : nil
        let exact = ExactComparison.compare(fused.out, stock.out)
        print(
            String(
                format: "M %d: stock %.4f ms, splitk-exact %.4f ms (%.2fx, %@), smallm-tile %@", m, stock.ms, fused.ms, stock.ms / fused.ms,
                exact.identical ? "bit-identical after \(steps) chained steps" : exact.description,
                tile.map { String(format: "%.4f ms (%.2fx)", $0.ms, stock.ms / $0.ms) } ?? "n/a (M > 256)"))
    }
    exit(0)
}

let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("arsenal-kernels-smoke-\(getpid())")
try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
var failed = false
@MainActor func check(_ name: String, _ ok: Bool, _ detail: String) { print("\(ok ? "PASS" : "FAIL") \(name): \(detail)"); if !ok { failed = true } }
func kernel(_ id: String, _ body: String, replaces: String) throws -> [String: Any] {
    let file = dir.appendingPathComponent("\(id).metal")
    try body.write(to: file, atomically: true, encoding: .utf8)
    return KernelSpec(
        id: id, mode: .exact, replaces: replaces, entry: id.replacingOccurrences(of: "-", with: "_"),
        files: [file.path], inputs: ["a", "b"], outputs: ["out"]
    ).configObject
}
// a * b + a in T, each op rounded to T like MLX's Multiply then Add.
let good = "uint i = thread_position_in_grid.x; if (i < uint(a_shape[0])) { T p = a[i] * b[i]; out[i] = p + a[i]; }"
let broken = "uint i = thread_position_in_grid.x; out[i] = no_such_function(a[i]);"
let kernels = try FolderKernels(config: [try kernel("muladd", good, replaces: "muladd"), try kernel("broken", broken, replaces: "broken")])

let n = 4099
let a = MLXRandom.normal([n], key: MLXRandom.key(1)).asType(.float16), b = MLXRandom.normal([n], key: MLXRandom.key(2)).asType(.float16)
@MainActor func run(_ site: String, _ on: Bool, threads: Int = 256) -> MLXArray {
    if on, let k = kernels.candidate(site),
        let y = k([a, b], template: [("T", DType.float16)], grid: (n, 1, 1), threadGroup: (threads, 1, 1), outputShapes: [[n]], outputDTypes: [.float16])?[0]
    {
        eval(y); return y
    }
    let y = a * b + a; eval(y); return y
}
let start = Date()
let gate = ExactGate.qualify(kernels: [kernels.candidate("muladd")!], model: "smoke", dtype: "fp16", weights: "-", checkRevision: "smoke-1") { [run("muladd", $0)] }
check("exact-gate", gate.pass, gate.detail + String(format: " (%.0f ms, first load)", Date().timeIntervalSince(start) * 1000))
kernels.activate("muladd", reason: gate.detail)
let again = ExactGate.qualify(kernels: [kernels.candidate("muladd")!], model: "smoke", dtype: "fp16", weights: "-", checkRevision: "smoke-1") { _ in [] }
check("exact-gate-persisted", again.pass && again.persisted, again.detail)
check(
    "folder-kernel-ran", kernels.active("muladd") != nil && kernels.candidate("muladd")!.preflightCompiles == 1,
    "preflight compiles \(kernels.candidate("muladd")!.preflightCompiles)")

let t0 = Date()
let fallback = run("broken", true)
let stock = a * b + a; eval(stock)
let reason = kernels.candidate("broken")?.failure ?? "none"
check(
    "compile-failure-falls-back", ExactComparison.compare(fallback, stock).identical && reason.contains("no_such_function"),
    String(format: "%@ (%.0f ms)", String(reason.prefix(140)), Date().timeIntervalSince(t0) * 1000))
// A gate whose "on" run fell back (compile failure) compares stock with stock: it must fail and save no pass.
let fellBack = ExactGate.qualify(kernels: [kernels.candidate("broken")!], model: "smoke", dtype: "fp16", weights: "-", checkRevision: "smoke-1") { [run("broken", $0)] }
check("exact-gate-refuses-fallback", !fellBack.pass && fellBack.detail.contains("preflight"), fellBack.detail)
let big = run("muladd", true, threads: 4096)
check(
    "oversized-threadgroup-falls-back", ExactComparison.compare(big, stock).identical && (kernels.candidate("muladd")?.failure ?? "").contains("exceeds"),
    kernels.candidate("muladd")?.failure ?? "no failure recorded")
let sum = (a + b).sum().item(Float.self)
check("mlx-healthy-after", sum.isFinite, "sum \(sum)")
// The preflight source must be the source MLX builds: capture MLX's own verbose print of the generated kernel.
let capture = dir.appendingPathComponent("verbose.txt")
FileManager.default.createFile(atPath: capture.path, contents: nil)
let handle = try FileHandle(forWritingTo: capture)
fflush(stdout); let saved = dup(1); dup2(handle.fileDescriptor, 1)
let probe = MLXFast.metalKernel(name: "muladd", inputNames: ["a", "b"], outputNames: ["out"], source: good)
let probed = probe(
    [a, b], template: [("T", DType.float16)], grid: (n, 1, 1), threadGroup: (256, 1, 1),
    outputShapes: [[n]], outputDTypes: [.float16], verbose: true)[0]
eval(probed)
fflush(stdout); dup2(saved, 1); close(saved)
let printed = (try? String(contentsOf: capture, encoding: .utf8)) ?? ""
let mlxSource = printed.components(separatedBy: "```\n").dropFirst().first.map { String($0.dropLast(1)) } ?? ""
let ours = MLXKernelSource.generate(
    name: "muladd", header: "", body: good, inputNames: ["a", "b"],
    inputs: [.init(type: "float16_t", ndim: 1, size: n), .init(type: "float16_t", ndim: 1, size: n)],
    outputNames: ["out"], outputTypes: ["float16_t"], template: [("T", .type("float16_t"))]
).source
check("source-matches-mlx", !mlxSource.isEmpty && mlxSource == ours, mlxSource.isEmpty ? "MLX printed nothing" : "\(ours.utf8.count) bytes, identical to MLX's generated source")

let states = kernels.states.map { "\($0.id)=\($0.active ? "active" : "stock")" }.joined(separator: " ")
check("states", kernels.states.allSatisfy { !$0.active }, states + " (muladd went stock after its oversized call: preflight failure disables the site)")
exit(failed ? 1 : 0)
