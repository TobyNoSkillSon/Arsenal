// SPDX-License-Identifier: MIT
import Foundation
import Metal
import Testing
import MLXPreamble
@testable import ArsenalKernelsCore
@testable import ArsenalKernels

// CPU/compile tests only: no MLXArray is created (the test build has no MLX metallib) and no kernel is dispatched.
// GPU behaviour (MLX runs the folder kernel, fallback after a failed preflight) is proven by arsenal-kernels-smoke.

private func temp(_ name: String) -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("arsenal-kernels-tests/\(name)-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

@Suite struct ResolutionTests {
    let declared = [
        KernelSpec(id: "a", chip: "*", mode: .exact, replaces: "linear", entry: "k", files: ["a.metal"], inputs: ["x"], outputs: ["y"]),
        KernelSpec(id: "b", chip: "m5", mode: .exact, replaces: "linear", entry: "k", files: ["b.metal"], inputs: ["x"], outputs: ["y"]),
        KernelSpec(id: "c", chip: "*", mode: .fast, replaces: "linear", entry: "k", files: ["c.metal"], inputs: ["x"], outputs: ["y"]),
        KernelSpec(id: "d", chip: "m4", mode: .fast, replaces: "rope", entry: "k", files: ["d.metal"], inputs: ["x"], outputs: ["y"]),
        KernelSpec(id: "e", chip: "*", mode: .exact, replaces: "rope", entry: "k", files: ["e.metal"], inputs: ["x"], outputs: ["y"])
    ]
    @Test func exactPrefersChipThenWildcard() {
        #expect(KernelSpec.resolve(declared, chipFamily: "m5", mode: .exact).map(\.id) == ["b", "e"])
        #expect(KernelSpec.resolve(declared, chipFamily: "m3", mode: .exact).map(\.id) == ["a", "e"])
    }
    @Test func fastIncludesExactWhereNoFastKernel() {
        #expect(KernelSpec.resolve(declared, chipFamily: "m5", mode: .fast).map(\.id) == ["c", "e"])
        #expect(KernelSpec.resolve(declared, chipFamily: "m4", mode: .fast).map(\.id) == ["c", "d"])
    }
    @Test func chipFamilyFromBrand() {
        #expect(KernelHost.family(brand: "Apple M5 Max") == "m5")
        #expect(KernelHost.family(brand: "Apple M1") == "m1")
        #expect(KernelHost.family(brand: "Intel(R) Core(TM) i9") == "unknown")
    }
}

@Suite struct ConfigTests {
    @Test func absentEmptyAndEntries() throws {
        #expect(try KernelSpec.parse(config: nil) == nil)
        #expect(try KernelSpec.parse(config: NSNull()) == nil)
        #expect(try KernelSpec.parse(config: [Any]()) == [])
        let spec = KernelSpec(
            id: "t", chip: "m5", mode: .fast, replaces: "linear.down", entry: "smallm_tile",
            files: ["/k/h.metal", "/k/b.metal"], header: "/k/h.metal", inputs: ["x", "w"], outputs: ["out"])
        let json = try JSONSerialization.data(withJSONObject: [spec.configObject])
        #expect(try KernelSpec.parse(json: json) == [spec])
        // The service sends absolute files and the manifest's relative header.
        var relative = spec.configObject; relative["header"] = "h.metal"
        #expect(try KernelSpec.parse(config: [relative])?.first?.header == "/k/h.metal")
    }
    @Test func rejectsMalformed() {
        #expect(throws: KernelError.self) { try KernelSpec.parse(config: ["x"]) }
        #expect(throws: KernelError.self) { try KernelSpec.parse(config: [["id": "a", "mode": "turbo", "replaces": "r", "entry": "k", "files": ["a.metal"]]]) }
        // mlx kernels need inputs/outputs; header must be one of the files; entry must be an identifier.
        #expect(throws: KernelError.self) { try KernelSpec.parse(config: [["id": "a", "mode": "exact", "replaces": "r", "entry": "k", "files": ["a.metal"]]]) }
        #expect(throws: KernelError.self) {
            try KernelSpec.parse(config: [["id": "a", "mode": "exact", "replaces": "r", "entry": "k", "files": ["a.metal"], "header": "h.metal", "inputs": ["x"], "outputs": ["y"]]]
            )
        }
        #expect(throws: KernelError.self) {
            try KernelSpec.parse(config: [["id": "a", "mode": "exact", "replaces": "r", "entry": "k-1", "files": ["a.metal"], "inputs": ["x"], "outputs": ["y"]]])
        }
    }
    @Test func loadsSourcesAndRejectsNonASCII() throws {
        let dir = temp("load")
        try "// header\n".write(to: dir.appendingPathComponent("h.metal"), atomically: true, encoding: .utf8)
        try "out[0] = x[0];\n".write(to: dir.appendingPathComponent("b.metal"), atomically: true, encoding: .utf8)
        try "// 2 × 2\n".write(to: dir.appendingPathComponent("u.metal"), atomically: true, encoding: .utf8)
        let spec = KernelSpec(id: "t", mode: .exact, replaces: "r", entry: "k", files: ["h.metal", "b.metal"], header: "h.metal", inputs: ["x"], outputs: ["out"])
        let loaded = try LoadedKernel.load(spec, base: dir)
        #expect(loaded.header == "// header\n" && loaded.body == "out[0] = x[0];\n")
        #expect(loaded.digest.count == 64)
        // A changed body changes the identity.
        let other = try LoadedKernel(spec: spec, header: loaded.header, body: "out[0] = x[1];\n")
        #expect(other.digest != loaded.digest)
        var bad = spec; bad.files = ["u.metal"]; bad.header = nil
        #expect(throws: KernelError.self) { try LoadedKernel.load(bad, base: dir) }
    }
}

@Suite struct MLXSourceTests {
    /// Golden output of mlx 0.32 `metal_kernel` (mlx-swift 9019419) for a templated two-input kernel, written out by
    /// hand from metal_kernel.cpp: shape buffer for `x` only (the body uses x_shape), a constant 1-element input,
    /// attributes in MLX's order, bool template printed as 1/0, the double underscore from make_template_hash.
    @Test func matchesMLXSignature() {
        let body = "const int n = x_shape[0];\nuint i = thread_position_in_grid.x;\nif (i < uint(n)) out[i] = x[i] * T(s[0]);"
        let (source, name) = MLXKernelSource.generate(
            name: "scale", header: "#include <metal_stdlib>\n", body: body, inputNames: ["x", "s"],
            inputs: [.init(type: "float16_t", ndim: 1, size: 1024), .init(type: "float", ndim: 1, size: 1)],
            outputNames: ["out"], outputTypes: ["float16_t"], template: [("T", .type("float16_t")), ("N", .int(4)), ("B", .bool(true))])
        #expect(name == "custom_kernel_scale__float16_t_4_1_float16_t_floatc_float16_t")
        let expected = """
            #include <metal_stdlib>
            template <typename T, int N, bool B>
            [[kernel]] void custom_kernel_scale__float16_t_4_1_float16_t_floatc_float16_t(
              const device float16_t* x [[buffer(0)]],
              const constant int* x_shape [[buffer(1)]],
              const constant float* s [[buffer(2)]],
              device float16_t* out [[buffer(3)]],
              uint3 thread_position_in_grid [[thread_position_in_grid]]) {
            \(body)
            }

            template [[host_name("custom_kernel_scale__float16_t_4_1_float16_t_floatc_float16_t")]] [[kernel]] decltype(custom_kernel_scale__float16_t_4_1_float16_t_floatc_float16_t<float16_t, 4, 1>) custom_kernel_scale__float16_t_4_1_float16_t_floatc_float16_t<float16_t, 4, 1>;

            """
        #expect(source == expected)
    }
    @Test func scalarInputAndNoTemplate() {
        let (source, name) = MLXKernelSource.generate(
            name: "k", header: "", body: "out[0] = a;", inputNames: ["a"],
            inputs: [.init(type: "float", ndim: 0, size: 1)], outputNames: ["out"], outputTypes: ["float"], template: [])
        #expect(name == "custom_kernel_k_floats_float")
        #expect(source.contains("const constant float& a [[buffer(0)]],\n  device float* out [[buffer(1)]]) {\nout[0] = a;\n}\n"))
        #expect(!source.contains("template"))
    }
}

@Suite(.serialized) struct CompileTests {
    static let preamble = String(cString: arsenal_mlx_metal_utils())
    func mlxSource(_ body: String, threads: Int = 64) -> (String, String) {
        let g = MLXKernelSource.generate(
            name: "probe", header: "", body: body, inputNames: ["x"], inputs: [.init(type: "float16_t", ndim: 1, size: 4096)],
            outputNames: ["out"], outputTypes: ["float16_t"], template: [("T", .type("float16_t"))])
        return (Self.preamble + g.source, g.kernelName)
    }

    @Test func preambleIsMLXs() {
        #expect(Self.preamble.contains("typedef bfloat bfloat16_t;"))
        #expect(Self.preamble.count > 10_000)
    }
    @Test func compilesValidKernelAndCachesLibrary() throws {
        let compiler = KernelCompiler()
        let (source, name) = mlxSource("uint i = thread_position_in_grid.x; out[i] = static_cast<T>(x[i] * T(2));")
        #expect(compiler.check(source: source, function: name, threadsPerThreadgroup: 256) == nil)
        #expect(compiler.check(source: source, function: name, threadsPerThreadgroup: 256) == nil)
        #expect(compiler.compileCount == 1)
    }
    @Test func reportsCompileErrorInsteadOfCrashing() {
        let compiler = KernelCompiler()
        let (source, name) = mlxSource("uint i = thread_position_in_grid.x; out[i] = undefined_symbol(x[i]);")
        let reason = compiler.check(source: source, function: name, threadsPerThreadgroup: 64)
        #expect(reason?.hasPrefix("Metal compile failed:") == true)
        #expect(reason?.contains("undefined_symbol") == true)
        // The failure is cached too: no second compile.
        _ = compiler.check(source: source, function: name, threadsPerThreadgroup: 64)
        #expect(compiler.compileCount == 1)
    }
    @Test func rejectsOversizedThreadgroup() {
        let (source, name) = mlxSource("uint i = thread_position_in_grid.x; out[i] = x[i];")
        let reason = KernelCompiler().check(source: source, function: name, threadsPerThreadgroup: 4096)
        #expect(reason?.contains("exceeds") == true)
    }
    /// The reference integration's folder kernels (models/laya-english/kernels) compile through the MLX preflight at
    /// the instantiations Laya uses (Metal 4 tensor ops: skipped on GPUs without them).
    @Test(arguments: [("smallm-tile", "m5/smallm_tile", [("SG", 8), ("SG", 4)]), ("splitk-exact", "any/splitk_exact", [("PS", 2048), ("P", 2)])])
    func layaFolderKernelsCompile(id: String, stem: String, ints: [(String, Int)]) throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("models/laya-english/kernels")
        guard MTLCreateSystemDefaultDevice()?.supportsFamily(.metal4) == true else { return }
        let spec = KernelSpec(
            id: id, mode: .fast, replaces: "linear.down", entry: id.replacingOccurrences(of: "-", with: "_"),
            files: [stem + ".h.metal", stem + ".metal"], header: stem + ".h.metal", inputs: ["x", "w", "bias", "res"], outputs: ["out"])
        let loaded = try LoadedKernel.load(spec, base: root)
        let template: [(String, MLXKernelSource.TemplateArg)] =
            id == "smallm-tile"
            ? [("T", .type("float16_t")), ("BM", .int(32)), ("BN", .int(32)), ("SG", .int(ints[0].1)), ("HAS_BIAS", .bool(true)), ("HAS_RES", .bool(true))]
            : [("T", .type("float16_t")), ("PS", .int(ints[0].1)), ("P", .int(ints[1].1)), ("HAS_BIAS", .bool(true)), ("HAS_RES", .bool(true))]
        let g = MLXKernelSource.generate(
            name: spec.entry, header: loaded.header, body: loaded.body, inputNames: spec.inputs,
            inputs: [
                .init(type: "float16_t", ndim: 2, size: 96 * 4096), .init(type: "float16_t", ndim: 2, size: 1024 * 4096),
                .init(type: "float16_t", ndim: 1, size: 1024), .init(type: "float16_t", ndim: 2, size: 96 * 1024)
            ],
            outputNames: ["out"], outputTypes: ["float16_t"], template: template)
        let reason = KernelCompiler().check(source: Self.preamble + g.source, function: g.kernelName, threadsPerThreadgroup: 64)
        #expect(reason == nil, "\(reason ?? "")")
    }
}

@Suite struct VerdictTests {
    @Test func persistsAndSeparatesKeys() {
        let file = temp("verdicts").appendingPathComponent("state/kernels.json")
        let store = KernelVerdicts(file: file)
        var runs = 0
        let key = KernelVerdicts.key("exact", "digest", "laya-english", "fp16", "check-1")
        let first = store.verdict(key) {
            runs += 1; return .init(pass: true, detail: "ok")
        }
        let second = KernelVerdicts(file: file).verdict(key) {
            runs += 1; return .init(pass: false, detail: "should not run")
        }
        #expect(!first.persisted && second.persisted && second.verdict.pass && runs == 1)
        // Another dtype is another verdict.
        let other = store.verdict(KernelVerdicts.key("exact", "digest", "laya-english", "int8", "check-1")) {
            runs += 1; return .init(pass: false, detail: "x")
        }
        #expect(!other.persisted && runs == 2)
        #expect(key.hasSuffix(KernelHost.key))
    }
    /// Keys are versioned: a pass saved by the previous schema (source-only digests, no engagement check) is never
    /// read and is dropped at the next write.
    @Test func olderSchemaPassesDoNotSurvive() throws {
        let file = temp("verdicts-legacy").appendingPathComponent("state/kernels.json")
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let legacyKey = (["exact", "digest", "laya-english", "fp16", "check-1"] + [KernelHost.key]).joined(separator: "|")
        try JSONEncoder().encode([legacyKey: KernelVerdicts.Verdict(pass: true, detail: "old")]).write(to: file)
        let store = KernelVerdicts(file: file)
        let key = KernelVerdicts.key("exact", "digest", "laya-english", "fp16", "check-1")
        #expect(key.hasPrefix("v2|") && key != legacyKey)
        var runs = 0
        _ = store.verdict(key) {
            runs += 1; return .init(pass: false, detail: "rechecked")
        }
        #expect(runs == 1)
        let saved = try JSONDecoder().decode([String: KernelVerdicts.Verdict].self, from: Data(contentsOf: file))
        #expect(saved.keys.sorted() == [key])
    }
    @Test func retentionEvictsOldestBeyondTheCap() throws {
        let file = temp("verdicts-cap").appendingPathComponent("state/kernels.json")
        let store = KernelVerdicts(file: file, limit: 3)
        for i in 0..<5 { store.write(KernelVerdicts.key("exact", "k\(i)"), .init(pass: true, detail: "", date: Double(100 + i))) }
        // A newest entry written with an old date is still kept (the entry being written always survives).
        store.write(KernelVerdicts.key("exact", "late"), .init(pass: true, detail: "", date: 1))
        let saved = try JSONDecoder().decode([String: KernelVerdicts.Verdict].self, from: Data(contentsOf: file))
        #expect(saved.count == 3)
        #expect(Set(saved.keys) == [KernelVerdicts.key("exact", "k3"), KernelVerdicts.key("exact", "k4"), KernelVerdicts.key("exact", "late")])
    }
}

/// A kernel as the gate sees it, with scripted dispatches and preflight result.
private final class FakeKernel: GatedKernel {
    let gateName = "fake"
    var calls = 0
    var failure: String?
}

@Suite struct GateEngagementTests {
    func store(_ name: String) -> KernelVerdicts { KernelVerdicts(file: temp(name).appendingPathComponent("state/kernels.json")) }

    @Test func passCountsWhenTheKernelRan() {
        let verdicts = store("gate-ran"), kernel = FakeKernel(), key = KernelVerdicts.key("exact", "ran")
        let first = verdicts.gate(key, kernels: [kernel]) {
            kernel.calls += 2; return .init(pass: true, detail: "identical")
        }
        #expect(first.verdict.pass && !first.persisted)
        var reran = false
        let second = verdicts.gate(key, kernels: [kernel]) {
            reran = true; return .init(pass: false, detail: "")
        }
        #expect(second.verdict.pass && second.persisted && !reran)
    }
    /// The caller fell back to its stock expression on every call: "on" equals "off", but nothing was proven.
    @Test func neverDispatchedNeverSavesAPass() {
        let verdicts = store("gate-idle"), kernel = FakeKernel(), key = KernelVerdicts.key("exact", "idle")
        let result = verdicts.gate(key, kernels: [kernel]) { .init(pass: true, detail: "identical") }
        #expect(!result.verdict.pass && result.verdict.detail.contains("never dispatched kernel fake"))
        #expect(verdicts.read(key)?.pass == false)
    }
    @Test func preflightFailureNeverSavesAPass() {
        let verdicts = store("gate-failed"), kernel = FakeKernel(), key = KernelVerdicts.key("exact", "failed")
        let result = verdicts.gate(key, kernels: [kernel]) {
            kernel.calls += 1; kernel.failure = "Kernel fake called with 0 inputs"
            return .init(pass: true, detail: "identical")
        }
        #expect(!result.verdict.pass && result.verdict.detail.contains("failed its preflight"))
        #expect(verdicts.read(key)?.pass == false)
    }
    @Test func everyKernelMustRun() {
        let verdicts = store("gate-two"), ran = FakeKernel(), idle = FakeKernel()
        let result = verdicts.gate(KernelVerdicts.key("exact", "two"), kernels: [ran, idle]) {
            ran.calls += 1; return .init(pass: true, detail: "")
        }
        #expect(!result.verdict.pass)
    }
}

@Suite struct IdentityTests {
    let spec = KernelSpec(
        id: "t", chip: "m5", mode: .exact, replaces: "linear.down", entry: "k", files: ["/k/h.metal", "/k/b.metal"],
        header: "/k/h.metal", inputs: ["x", "w"], outputs: ["out", "aux"])
    func digest(_ spec: KernelSpec, header: String = "// h\n", body: String = "out[0] = x[0] - w[0];") throws -> String {
        try LoadedKernel(spec: spec, header: header, body: body).digest
    }
    /// Every declaration MLX or the runner binds by is part of the identity: a descriptor-only edit is a cache miss.
    @Test func descriptorEditsChangeTheIdentity() throws {
        let base = try digest(spec)
        var edits: [KernelSpec] = []
        var s = spec; s.inputs = ["w", "x"]; edits.append(s)
        s = spec; s.outputs = ["aux", "out"]; edits.append(s)
        s = spec; s.inputs = ["x", "w", "b"]; edits.append(s)
        s = spec; s.entry = "k2"; edits.append(s)
        s = spec; s.replaces = "linear.up"; edits.append(s)
        s = spec; s.mode = .fast; edits.append(s)
        s = spec; s.api = .metal; edits.append(s)
        let digests = try edits.map { try digest($0) }
        #expect(!digests.contains(base))
        #expect(Set(digests).count == digests.count)
        // Moving text between header and body changes the generated source (the signature sits between them).
        #expect(try digest(spec, header: "", body: "// h\nout[0] = x[0] - w[0];") != base)
        // Names cannot forge a field boundary.
        var joined = spec; joined.inputs = ["xw"]
        var split = spec; split.inputs = ["x", "w"]
        #expect(try digest(joined) != digest(split))
    }
    /// Labels do not change what runs: the same kernel under another id or chip keeps its verdicts.
    @Test func labelsKeepTheIdentity() throws {
        var s = spec; s.id = "renamed"; s.chip = "*"
        #expect(try digest(s) == digest(spec))
    }
}

@Suite struct PreflightFailureTests {
    /// An argument mismatch is a preflight failure like a compile error: `failure` records it, so the site goes stock
    /// and a gate knows its kernel-on run fell back. (No MLXArray is created: zero inputs.)
    @Test func argumentMismatchSetsFailure() throws {
        let spec = KernelSpec(id: "t", mode: .exact, replaces: "r", entry: "k", files: ["/b.metal"], inputs: ["x"], outputs: ["out"])
        let kernel = try FolderMLXKernel(
            try LoadedKernel(spec: spec, header: "", body: "out[0] = x[0];"),
            verdicts: KernelVerdicts(file: temp("preflight").appendingPathComponent("kernels.json")))
        let reason = kernel.preflight(inputs: [], template: [], outputDTypes: [], threadGroup: (1, 1, 1))
        #expect(reason?.contains("declares 1 / 1") == true)
        #expect(kernel.failure == reason)
        #expect(kernel.calls == 0)
    }
}

@Suite struct ExactTests {
    func data(_ values: [UInt16]) -> Data { values.withUnsafeBufferPointer { Data(buffer: $0) } }
    @Test func identicalBuffers() {
        let a = data([0x3C00, 0x4000, 0x0000])
        #expect(ExactComparison.compare(a, a, element: .float16).identical)
    }
    @Test func oneULPIsADifference() {
        let r = ExactComparison.compare(data([0x3C00, 0x4000]), data([0x3C00, 0x4001]), element: .float16)
        #expect(!r.identical && r.mismatches == 1 && r.firstMismatch == 1 && r.maxAbsDiff > 0 && r.maxAbsDiff < 0.01)
    }
    @Test func signedZeroAndNaNPayloadsDiffer() {
        #expect(!ExactComparison.compare(data([0x0000]), data([0x8000]), element: .float16).identical)
        #expect(!ExactComparison.compare(data([0x7E00]), data([0x7E01]), element: .float16).identical)
        #expect(ExactComparison.compare(data([0x7E00]), data([0x7E00]), element: .float16).identical)
    }
    @Test func bfloat16AndFloat32Decode() {
        let r = ExactComparison.compare(data([0x3F80]), data([0x4000]), element: .bfloat16) // 1.0 vs 2.0
        #expect(r.maxAbsDiff == 1)
        let f = [Float(1), 3].withUnsafeBufferPointer { Data(buffer: $0) }, g = [Float(1), 2.5].withUnsafeBufferPointer { Data(buffer: $0) }
        #expect(ExactComparison.compare(f, g, element: .float32).maxAbsDiff == 0.5)
    }
    @Test func layoutMismatch() {
        #expect(ExactComparison.compare(data([1, 2]), data([1]), element: .float16).layoutMismatch != nil)
    }
}
