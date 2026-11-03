import Foundation
import MLX
import MLXNN
import VerdictSmallMGEMM

/// Verdict's use of the family small-M GEMM (`SmallMGEMM`, vendored verbatim from Vella) for single items and other
/// small requests: a single item is 2 rows × ~48 tokens → M ≈ 96 after the 16-token bucket, where MLX 0.32.2's NAX
/// GEMM (64 × 128 tiles) leaves most GPU cores idle. Dense fp16/bf16 Linears only, rows in `rows`; quantized
/// precisions and big (batched) chunks stay on MLX. Its split-K sums are not bit-identical to MLX, so every use is
/// gated (see `SmallMGate`) and reported in /status (`optimizations.smallm`).
public enum SmallMRoute {
    static let environment = ProcessInfo.processInfo.environment
    /// Whether the component is requested for a network whose default is `defaultOn` (Laya: on since 28 Sep, after
    /// the fixture gate and an A/B on single items; Von: off, single items gained < 5 %). `VERDICT_SMALLM=1`
    /// / `=0` overrides the default; `VERDICT_STOCK_PATH=1` always turns it off. Changing a default needs a
    /// `SmallMGate.version` bump.
    static func requested(defaultOn: Bool) -> Bool {
        if environment["VERDICT_STOCK_PATH"] == "1" { return false }
        switch environment["VERDICT_SMALLM"] {
        case "1": return true
        case "0": return false
        default: return defaultOn
        }
    }
    /// Rows (M = batch rows × tokens) routed to the kernel: the library's tile range. Up to 8 rows MLX's gemv is
    /// faster; the routed K > N shapes still win at M 256 (2624→1024 1.76×, 4096→1024 1.6×, M5 Max fp16). Verdict's
    /// single items are M 96–128 (job ads, 2 questions), 192 (20-option choice) and 128–336 (passages).
    /// `VERDICT_SMALLM_MAX_ROWS` narrows it for A/B.
    static let rows: ClosedRange<Int> = 9 ... max(9, min(SmallMGEMM.tileRows.upperBound, Int(environment["VERDICT_SMALLM_MAX_ROWS"] ?? "") ?? SmallMGEMM.tileRows.upperBound))
    /// Residual adds folded into the GEMM epilogue (added in float before the one rounding: one kernel fewer per
    /// residual Linear, not bit-identical to matmul-then-add). `VERDICT_SMALLM_RESIDUAL=0` keeps the separate add.
    static let residualEpilogue = environment["VERDICT_SMALLM_RESIDUAL"] != "0"
    /// Which Linears: "down" (default) only K > N — the down/contracting projections, where MLX gets few tiles for a
    /// long K (M 96, fp16, M5 Max: 2624→1024 1.68×, 4096→1024 1.5×, 1152→768 1.12× per GEMM) — while K ≤ N shapes
    /// tie or lose (1024→3072 0.8×, 1024→1024 1.0×, 1024→5248 1.0×, measured per GEMM). "all" routes every
    /// eligible Linear (A/B). `VERDICT_SMALLM_SHAPES`.
    static let shapes = environment["VERDICT_SMALLM_SHAPES"] ?? "down"
    static var allShapes: Bool { shapes == "all" }
    /// Whether a K → N Linear is routed under `shapes` ("square": K ≥ N, adds the 1024² attention output).
    static func routes(k: Int, n: Int) -> Bool { shapes == "all" || k > n || (shapes == "square" && k == n) }
    /// Test-only: VERDICT_TEST_SMALLM_FAULT=nan poisons served outputs while the kernel is on (never the gate's own
    /// runs), to exercise the runtime fallback. Part of the gate key through `switchKey`.
    public static let injectNaN = environment["VERDICT_TEST_SMALLM_FAULT"] == "nan"
    /// The value that selects this component; part of the gate key (an override run cannot qualify the default).
    static var switchKey: String {
        "VERDICT_SMALLM=\(environment["VERDICT_SMALLM"] ?? "default") residual=\(residualEpilogue) rows=\(rows.upperBound) shapes=\(shapes) fault=\(injectNaN)"
    }

    /// y = x·Wᵀ (+ bias) (+ residual) on the small-M kernel, or nil (caller runs its stock expression). The residual
    /// is only folded when `residualEpilogue`; otherwise the caller adds it.
    static func linear(_ x: MLXArray, _ layer: Linear, residual: MLXArray? = nil) -> MLXArray? {
        // QuantizedLinear subclasses Linear: exact type only.
        guard type(of: layer) == Linear.self else { return nil }
        let w = layer.weight
        guard w.ndim == 2, x.dtype == w.dtype, x.dtype == .float16 || x.dtype == .bfloat16 else { return nil }
        let k = x.dim(-1), n = w.dim(0), m = x.size / max(k, 1)
        guard rows.contains(m), w.dim(1) == k, routes(k: k, n: n) else { return nil }
        let bias = layer.bias.flatMap { $0.dtype == x.dtype ? $0 : nil }
        if layer.bias != nil && bias == nil { return nil }
        let r = residualEpilogue ? residual.flatMap { $0.dtype == x.dtype && $0.size == m * n ? $0.reshaped([m, n]) : nil } : nil
        let epilogue: SmallMGEMM.Epilogue
        switch (bias, r) {
        case let (b?, r?): epilogue = .biasResidual(b, r)
        case let (b?, nil): epilogue = .bias(b)
        case let (nil, r?): epilogue = .residual(r)
        case (nil, nil): epilogue = .none
        }
        guard SmallMGEMM.supports(m: m, n: n, k: k, dtype: x.dtype, format: .dense, epilogue: epilogue.kind) else { return nil }
        guard let y = SmallMGEMM.matmul(x, .dense(w), epilogue: epilogue) else { return nil }
        // Residual requested but not folded: the caller's contract is residual + xWᵀ.
        if let residual, r == nil { return residual + y }
        return y
    }

    /// Verdict's own kernel check (run in the gate child, never in the serving process first): every (K, N) Verdict
    /// routes, at both simdgroup classes (M 16 and 96) and with the epilogue each site uses, against stock MLX.
    /// Returns relative RMS per case (∞ when non-finite or unsupported where Verdict expects support).
    public static func verdictShapesCheck() -> [String: Float] {
        var out: [String: Float] = [:]
        func random(_ shape: [Int], _ seed: UInt64, scale: Float = 1) -> MLXArray {
            (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(.float16)
        }
        // (K, N, bias, residual): Laya EN/Typed + Von encoder, Laya heads, Laya Multilingual.
        let shapes: [(Int, Int, Bool, Bool)] = [(1024, 3072, false, false), (1024, 1024, false, true), (1024, 5248, false, false), (2624, 1024, false, true),
                                                (1024, 3072, true, false), (1024, 1024, true, true), (1024, 4096, true, false), (4096, 1024, true, true),
                                                (768, 2304, false, false), (768, 768, false, true), (1152, 768, false, true)]
        var seed: UInt64 = 0x7e4d
        for m in [16, 96, 256] {
            for (k, n, hasBias, hasResidual) in shapes {
                seed += 3
                let x = random([m, k], seed), w = random([n, k], seed + 1, scale: 0.03)
                let b: MLXArray? = hasBias ? random([n], seed + 2, scale: 0.1) : nil
                let res: MLXArray? = hasResidual ? random([m, n], seed + 2) : nil
                var reference = MLX.matmul(x, w.transposed())
                if let b { reference = reference + b }
                if let res { reference = res + reference }
                let layer = Linear(weight: w, bias: b)
                if !routes(k: k, n: n) { continue }   // not routed
                let name = "k\(k).n\(n)\(hasBias ? ".bias" : "")\(hasResidual ? ".res" : "").m\(m)"
                if let y = linear(x, layer, residual: res) {
                    out[name] = SmallMGEMM.relativeRMS(y, reference)
                } else {
                    out[name] = .infinity
                }
            }
        }
        return out
    }
}

/// Persisted verdicts for the small-M component, keyed by everything that changes what runs: the switch values,
/// the kernel revision, this gate's version, GPU architecture, OS build, the helper binary (size + mtime) and, for the
/// model stage, model id (with the weights' identity) and precision. File: <support>/smallm-gate.json, bounded: entries of
/// another gate version are dropped and at most `limit` are kept (oldest first out).
public enum SmallMGate {
    /// Bump when the gate's checks or thresholds change. gate-3: a pass needs the folder kernel to have run.
    static let version = "gate-3"
    static let limit = 64
    /// Relative RMS bound for the kernel checks (per GEMM; split-K fp16 measured ~1e-3 on Parakeet, injected bugs 0.46+).
    public static let kernelBound: Float = 2e-2
    /// Model stage: relative RMS of a model's marker logits with vs without the kernel over the built-in items.
    public static let logitsBound: Float = 1e-2
    /// Model stage: rows whose stock top-two logits are closer than this are near-ties (counted, not judged).
    public static let tieMargin: Float = 0.05

    // Arsenal: the gate verdicts live in Arsenal's state dir (ARSENAL_SUPPORT_DIR), never in Verdict's.
    static var supportDirectory: URL {
        let env = ProcessInfo.processInfo.environment
        let support = env["ARSENAL_SUPPORT_DIR"] ?? ((env["HOME"] ?? NSHomeDirectory()) + "/Library/Application Support/Arsenal")
        return URL(fileURLWithPath: support, isDirectory: true).appendingPathComponent("state", isDirectory: true)
    }
    static var file: URL { supportDirectory.appendingPathComponent("smallm-gate.json") }
    static let lock = NSLock()

    static var baseKey: String {
        var build = [CChar](repeating: 0, count: 64), size = build.count
        let os = sysctlbyname("kern.osversion", &build, &size, nil, 0) == 0 ? String(cString: build) : "?"
        let exe = URL(fileURLWithPath: CommandLine.arguments.first ?? "")
        let attrs = try? FileManager.default.attributesOfItem(atPath: exe.path)
        let size_ = (attrs?[.size] as? NSNumber)?.intValue ?? 0
        let mtime = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return [version, SmallMGEMM.revision, SmallMRoute.switchKey, GPU.deviceInfo().architecture, os, "\(exe.lastPathComponent):\(size_):\(Int(mtime))"].joined(separator: "|")
    }

    static func read(_ key: String) -> [String: Any]? {
        lock.lock(); defer { lock.unlock() }
        guard let data = try? Data(contentsOf: file), let all = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return all[key] as? [String: Any]
    }
    static func write(_ key: String, _ value: [String: Any]) {
        lock.lock(); defer { lock.unlock() }
        var all = (try? Data(contentsOf: file)).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        all = all.filter { $0.key.contains("|" + version + "|") }
        all[key] = value
        if all.count > limit {
            func date(_ v: Any) -> Double { (v as? [String: Any])?["date"] as? Double ?? 0 }
            let evict = all.filter { $0.key != key }.sorted { (date($0.value), $0.key) < (date($1.value), $1.key) }
            for (old, _) in evict.prefix(all.count - limit) { all[old] = nil }
        }
        try? FileManager.default.createDirectory(at: supportDirectory, withIntermediateDirectories: true)
        if let data = try? JSONSerialization.data(withJSONObject: all, options: [.sortedKeys]) {
            let tmp = supportDirectory.appendingPathComponent("smallm-gate.\(getpid()).tmp")
            if (try? data.write(to: tmp)) != nil { _ = rename(tmp.path, file.path) }
        }
    }

    /// Stage 0: the kernel checks, once per key. nil = pass. Arsenal: in process. Verdict ran them in a child because a
    /// Metal compile error inside MLX aborts the process; the folder kernel (ArsenalKernels.FolderMLXKernel) compiles
    /// every instantiation through Metal first and answers nil instead, so a bad source costs a stock path, not the runner.
    static func kernelVerdict() -> String? {
        let key = "kernel|" + baseKey
        if let saved = read(key) { return saved["pass"] as? Bool == true ? nil : (saved["reason"] as? String ?? "kernel check failed") }
        let reason = engaged(inProcessKernelCheck())
        write(key, ["pass": reason == nil, "reason": reason.map { $0 as Any } ?? NSNull(), "date": Date().timeIntervalSince1970])
        return reason
    }

    /// A check's reason, or why its pass does not count: the folder tile kernel failed a preflight or was never
    /// dispatched since `since` (every call fell back to stock MLX, so "with the kernel" measured stock).
    static func engaged(_ reason: String?, since: Int = 0) -> String? {
        if let reason { return reason }
        guard let tile = SmallMGEMM.tile else { return "no folder tile kernel" }
        if let failure = tile.failure { return "folder kernel preflight: \(failure)" }
        return tile.calls > since ? nil : "the check never dispatched the folder kernel"
    }

    /// The kernel checks themselves (what the child runs): library self-test + Verdict's shapes. nil = pass.
    public static func inProcessKernelCheck() -> String? {
        let library = SmallMGEMM.selfTest()
        let failed = SmallMGEMM.selfTestFailures(library)
        if !failed.isEmpty { return "library self-test: " + failed.joined(separator: ", ") }
        let ours = SmallMRoute.verdictShapesCheck()
        let bad = ours.filter { !($0.value <= kernelBound) }.keys.sorted()
        if !bad.isEmpty { return "Verdict shapes: " + bad.prefix(4).joined(separator: ", ") }
        return nil
    }

    /// Stages 1 + 2 for one loaded model: `check` runs the model with and without the kernel and returns nil (pass,
    /// with a detail string) or the failure reason. Persisted per model + precision.
    static func modelVerdict(model: String, bits: Int, check: () -> (pass: Bool, detail: String)) -> (pass: Bool, detail: String) {
        let key = "model|\(model)|\(bits)|" + baseKey
        if let saved = read(key), let pass = saved["pass"] as? Bool { return (pass, (saved["detail"] as? String ?? "") + ", persisted") }
        let before = SmallMGEMM.tile?.calls ?? 0
        var result = check()
        if result.pass, let refused = engaged(nil, since: before) { result = (false, refused) }
        write(key, ["pass": result.pass, "detail": result.detail, "date": Date().timeIntervalSince1970])
        return result
    }
}

/// One network's small-M component: gate verdict, stock-path switch and runtime fallback, and its /status line.
final class SmallMComponent {
    private(set) var validated = false
    /// The model serves on its stock path (helper fallback or VERDICT_STOCK_PATH).
    var stockForced = false
    /// Switched off after non-finite output on it (until reload).
    private(set) var runtimeOff = false
    private var detail = "", offReason = "off"
    var active: Bool { validated && !stockForced && !runtimeOff }
    /// /status optimizations.smallm. Contains "small" only while active (the catalog's stock guard looks for it).
    var path: String {
        if active { return detail }
        if runtimeOff { return "stock matmul (switched off after non-finite output)" }
        return validated ? "stock matmul (stock path)" : offReason
    }
    func disable() { runtimeOff = true }
    /// The gate (SmallMGate): eligibility, kernel stage (child process, persisted), then `modelCheck` (outputs with vs
    /// without the kernel + chosen answers on built-in items, persisted per model + precision). A failure disables only
    /// this component. `ineligible`: why this model/precision cannot use it (nil = eligible).
    /// `allowed`: this model's recipe includes inexact kernels (Optimized · Fast); Standard and Exact never run it.
    func qualify(model: String, bits: Int, defaultOn: Bool, allowed: Bool = true, ineligible: String?, modelCheck: () -> (pass: Bool, detail: String)) {
        guard allowed else { offReason = "off (recipe without inexact kernels)"; return }
        guard SmallMRoute.requested(defaultOn: defaultOn) else { offReason = "off"; return }
        if let ineligible { offReason = "stock matmul (\(ineligible))"; return }
        guard SmallMGEMM.tile != nil else { offReason = "stock matmul (the model folder declares no smallm_tile kernel for this chip)"; return }
        guard SmallMGEMM.tensorOpsAvailable else { offReason = "stock matmul (needs an M5-class GPU)"; return }
        if let reason = SmallMGate.kernelVerdict() { offReason = "stock matmul (kernel check failed: \(reason.prefix(120)))"; return }
        let verdict = SmallMGate.modelVerdict(model: model, bits: bits, check: modelCheck)
        Memory.clearCache()
        guard verdict.pass else { offReason = "stock matmul (model check failed: \(verdict.detail.prefix(160)))"; return }
        validated = true
        detail = "small-M GEMM (\(SmallMGEMM.revision), \(SmallMRoute.shapes == "all" ? "all shapes" : SmallMRoute.shapes == "square" ? "K>=N shapes" : "K>N shapes"), rows \(SmallMRoute.rows.lowerBound)-\(SmallMRoute.rows.upperBound)"
            + (SmallMRoute.residualEpilogue ? ", residual epilogue" : "") + "; \(verdict.detail))"
    }
}
