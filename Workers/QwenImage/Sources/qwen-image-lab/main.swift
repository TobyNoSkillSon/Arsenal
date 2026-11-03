// SPDX-License-Identifier: MIT
// Lab-only Qwen-Image runner: standalone real-weight generations and component checks on fixed inputs, with an MLX
// memory limit and a whole-Mac RAM admission check. Reads a pinned Hugging Face snapshot (BF16, plus the base VAE's
// FP32 file, cast to the selected half at load: lab only; Arsenal's Get stores that cast). Never bundled.
import Darwin
import Foundation
import IntegrationKit
import MLX
import MLXNN
import MLXRandom
import ModelPrecision
import QwenImageNative

struct LabError: Error, CustomStringConvertible { let description: String; init(_ d: String) { description = d } }

NativeRuntime.prepare()
var arguments = Array(CommandLine.arguments.dropFirst())
let mode = arguments.isEmpty ? "help" : arguments.removeFirst()
var options: [String: [String]] = [:]
var flags = Set<String>()
while let first = arguments.first {
    arguments.removeFirst()
    guard first.hasPrefix("--") else { continue }
    let key = String(first.dropFirst(2))
    if let next = arguments.first, !next.hasPrefix("--") { options[key, default: []].append(next); arguments.removeFirst() } else { flags.insert(key) }
}
func option(_ key: String) -> String? { options[key]?.last }
func number(_ key: String, _ fallback: Int) -> Int { option(key).flatMap(Int.init) ?? fallback }

// MARK: Memory

/// Whole-Mac RAM in use: (active + wired + compressor) pages, as vm_stat reports them.
func macInUseBytes() -> Int {
    var stats = vm_statistics64_data_t(), count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
    }
    guard status == KERN_SUCCESS else { return Int.max }
    return (Int(stats.active_count) + Int(stats.wire_count) + Int(stats.compressor_page_count)) * Int(vm_kernel_page_size)
}
func swapUsedBytes() -> Int {
    var usage = xsw_usage(), size = MemoryLayout<xsw_usage>.size
    return sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 ? Int(usage.xsu_used) : 0
}
func footprintPeakBytes() -> Int {
    var info = task_vm_info_data_t(), count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return status == KERN_SUCCESS ? Int(info.ledger_phys_footprint_peak) : 0
}
let gb = 1_000_000_000.0
let startSwap = swapUsedBytes()
/// Refuses to load when the Mac would pass 95 GB in use, and stops if swap grew by more than 1 GB since start.
func admit(_ label: String, expectedBytes: Int) throws {
    let inUse = macInUseBytes()
    FileHandle.standardError.write(
        Data(String(format: "[%@] Mac in use %.1f GB, expected +%.1f GB, swap %.2f GB\n", label, Double(inUse) / gb, Double(expectedBytes) / gb, Double(swapUsedBytes()) / gb).utf8)
    )
    guard inUse + expectedBytes < 95_000_000_000 else {
        throw LabError("\(label): \(String(format: "%.1f", Double(inUse) / gb)) GB in use + \(String(format: "%.1f", Double(expectedBytes) / gb)) GB would pass 95 GB")
    }
    try checkSwap(label)
}
func checkSwap(_ label: String) throws {
    let grown = swapUsedBytes() - startSwap
    if grown > 1_000_000_000 { throw LabError("\(label): swap grew \(String(format: "%.2f", Double(grown) / gb)) GB; stopping") }
}

// MARK: Loading a raw snapshot

let weights = URL(fileURLWithPath: option("weights") ?? ".")
let dtype = option("dtype") ?? "bf16"
let half: DType = option("half") == "fp16" || dtype == "fp16" ? .float16 : .bfloat16
func json(_ path: String) throws -> [String: Any] { try ConfigFile.json(weights.appendingPathComponent(path)) }
func archives(_ directory: String) throws -> [URL] {
    try FileManager.default.contentsOfDirectory(at: weights.appendingPathComponent(directory), includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
}
func tier(_ precision: String) throws -> NativePrecisionTier { try NativePrecisionTier(precision, half: half) }
/// Lab only: the raw base snapshot keeps an FP32 VAE; Get stores it in the served half.
func labHalf(_ value: MLXArray) -> MLXArray { value.dtype == .float32 ? value.asType(half) : value }

/// `--encoder-weights` reads the prompt encoder from another snapshot (lab: Turbo's encoder before it finished downloading).
let encoderRoot = URL(fileURLWithPath: option("encoder-weights") ?? weights.path)
func loadEncoder(_ precision: String) throws -> Qwen3VLEncoder {
    let encoder = Qwen3VLEncoder(try Qwen3VLConfig(json: ConfigFile.json(encoderRoot.appendingPathComponent("text_encoder/config.json"))))
    let files = try FileManager.default.contentsOfDirectory(at: encoderRoot.appendingPathComponent("text_encoder"), includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    _ = try tier(precision).load(
        encoder, archives: files, map: { k, v in QwenImageWeights.encoderName(k).map { ($0, labHalf(v)) } }, eligible: QwenImageWeights.eligible)
    encoder.train(false); return encoder
}
func loadTransformer(_ precision: String) throws -> QwenImageTransformer {
    let transformer = QwenImageTransformer(try TransformerConfig(json: json("transformer/config.json")))
    _ = try tier(precision).load(
        transformer, archives: archives("transformer"), map: { k, v in (QwenImageWeights.transformerName(k), labHalf(v)) },
        eligible: QwenImageWeights.eligible)
    transformer.train(false); return transformer
}
func loadVAE(_ precision: String) throws -> QwenImageVAE {
    let vae = QwenImageVAE(try VAEConfig(json: json("vae/config.json")))
    _ = try tier(precision).load(
        vae, archives: archives("vae"), map: { k, v in QwenImageWeights.vaeName(k).map { ($0, QwenImageWeights.vaeTensor($0, labHalf(v))) } },
        eligible: { _, _ in false })
    vae.train(false); return vae
}
func tokenizer() throws -> (String) -> [Int] {
    let t = try LocalTokenizer.load(weights.appendingPathComponent("processor"))
    return { t.encode(text: $0, addSpecialTokens: false) }
}
func timed<T>(_ body: () throws -> T) rethrows -> (T, Double) { let s = Date(); let v = try body(); return (v, Date().timeIntervalSince(s)) }

func request() throws -> GenerationRequest {
    var r = GenerationRequest(prompt: option("prompt") ?? "A lighthouse on a cliff at sunset, oil painting")
    r.width = option("width").flatMap(Int.init); r.height = option("height").flatMap(Int.init)
    r.steps = number("steps", 40); r.seed = UInt64(number("seed", 42))
    r.negativePrompt = option("negative"); r.guidance = Float(option("guidance") ?? "1") ?? 1
    r.images = try (options["image"] ?? []).map { try ImageOps.read(URL(fileURLWithPath: $0)) }
    if r.images.isEmpty, r.width == nil { r.width = 1024; r.height = 1024 }
    return r
}

// MARK: Modes

func generate() throws {
    Memory.memoryLimit = number("limit-gb", 60) * 1_000_000_000
    let expect = Int((Double(option("expect-gb") ?? "40") ?? 40) * gb)
    try admit("load", expectedBytes: expect)
    let sequential = flags.contains("sequential")
    let r = try request()
    let scheduler = try SchedulerConfig(json: json("scheduler/scheduler_config.json"), pipeline: json("model_index.json"))
    var report: [String: Any] = [
        "dtype": dtype, "sequential": sequential, "prompt": r.prompt, "seed": r.seed, "steps_requested": r.steps,
        "images": options["image"] ?? [], "chip": "M5 Max", "date": ISO8601DateFormatter().string(from: Date())
    ]
    let total = Date()
    Memory.peakMemory = 0
    // No local reference to the encoder: --sequential must be able to free it.
    let pipeline = QwenImagePipeline(encoder: nil, transformer: nil, vae: nil, tokenize: try tokenizer(), scheduler: scheduler)
    var encoderLoad = 0.0, transformerLoad = 0.0, vaeLoad = 0.0
    (pipeline.encoder, encoderLoad) = try timed { try loadEncoder(dtype) }
    if !sequential {
        (pipeline.transformer, transformerLoad) = try timed { try loadTransformer(dtype) }
        (pipeline.vae, vaeLoad) = try timed { try loadVAE(dtype) }
    }
    try checkSwap("loaded")
    let residentAfterLoad = Memory.activeMemory
    let (conditioning, encode) = try timed { try pipeline.condition(r) }
    let encoderPeak = Memory.peakMemory
    if sequential {
        pipeline.encoder = nil; Memory.clearCache()
        (pipeline.transformer, transformerLoad) = try timed { try loadTransformer(dtype) }
        (pipeline.vae, vaeLoad) = try timed { try loadVAE(dtype) }
    }
    let schedule = try pipeline.schedule(r)
    var stepTimes: [Double] = [], last = Date()
    let (image, render) = try timed {
        try pipeline.render(conditioning, r) { step, steps in
            stepTimes.append(Date().timeIntervalSince(last)); last = Date()
            FileHandle.standardError.write(Data("step \(step)/\(steps) \(String(format: "%.2f", stepTimes.last!)) s\n".utf8))
            try checkSwap("step \(step)")
        }
    }
    let out = URL(fileURLWithPath: option("out") ?? "qwen-image.png")
    try ImageOps.writePNG(image, to: out)
    let t = pipeline.lastTimings
    report["out"] = out.path; report["width"] = image.width; report["height"] = image.height; report["alpha"] = image.transparent
    report["sigmas"] = schedule.sigmas.map(Double.init)
    report["seconds"] = [
        "load_encoder": encoderLoad, "load_transformer": transformerLoad, "load_vae": vaeLoad, "encode": encode, "prefill": t.prefill, "denoise": t.denoise,
        "decode": t.decode, "render": render, "generation": encode + render, "total": Date().timeIntervalSince(total),
        "step_median": stepTimes.sorted()[stepTimes.count / 2], "steps": stepTimes
    ]
    report["memory_gb"] = [
        "mlx_peak": Double(Memory.peakMemory) / gb, "mlx_encoder_phase_peak": Double(encoderPeak) / gb, "mlx_resident_after_load": Double(residentAfterLoad) / gb,
        "footprint_peak": Double(footprintPeakBytes()) / gb, "mac_in_use_end": Double(macInUseBytes()) / gb, "swap_growth": Double(swapUsedBytes() - startSwap) / gb
    ]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    if let path = option("report") { try data.write(to: URL(fileURLWithPath: path)) }
    print(String(decoding: data, as: UTF8.self))
}

/// Saves fixed component inputs/outputs once (`--capture`), or compares a candidate tier against a saved capture.
/// Encoder: last-layer states for a fixed prompt. Transformer: velocity at three timesteps for saved states and fixed
/// noise. VAE: decode of a fixed latent. Reports relative L2, cosine, max |x| and finiteness.
func components() throws {
    Memory.memoryLimit = number("limit-gb", 60) * 1_000_000_000
    let dir = URL(fileURLWithPath: option("dir") ?? "components")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let tag = option("tag") ?? dtype
    let component = option("component") ?? "all"
    var outputs: [String: MLXArray] = [:]
    let inputsURL = dir.appendingPathComponent("inputs.safetensors")
    // Materialised now: the file is rewritten below, and lazily loaded arrays would read it afterwards.
    var inputs = (try? loadArrays(url: inputsURL)) ?? [:]
    eval(Array(inputs.values))
    let prompt = option("prompt") ?? "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
    if component == "all" || component == "encoder" {
        try admit("encoder \(tag)", expectedBytes: Int((Double(option("expect-gb") ?? "20") ?? 20) * gb))
        let encoder = try loadEncoder(dtype)
        let pipeline = QwenImagePipeline(encoder: encoder, transformer: nil, vae: nil, tokenize: try tokenizer(), scheduler: SchedulerConfig())
        var r = GenerationRequest(prompt: prompt); r.width = 1024; r.height = 1024
        let c = try pipeline.condition(r)
        outputs["encoder.states"] = c.positive.states.asType(.float32)
        // Text-only prompt: no image slots to save.
        if inputs["text.states"] == nil { inputs["text.states"] = c.positive.states.asType(.float32) }
        Memory.clearCache()
    }
    if component == "all" || component == "transformer" {
        try admit("transformer \(tag)", expectedBytes: Int((Double(option("expect-gb") ?? "18") ?? 18) * gb))
        guard let states = inputs["text.states"] else { throw LabError("Capture the encoder first") }
        let transformer = try loadTransformer(dtype)
        var residual: [String: Float] = [:]
        transformer.residualObserver = { phase, _, h in
            let peak = abs(h.asType(.float32)).max().item(Float.self)
            residual[phase] = max(residual[phase] ?? 0, peak.isFinite ? peak : .infinity)
        }
        defer { FileHandle.standardError.write(Data("pre-clip residual max |h|: \(residual)\n".utf8)) }
        let size = number("size", 512) / 16
        let noiseKey = size == 32 ? "noise" : "noise\(size)"
        if inputs[noiseKey] == nil { inputs[noiseKey] = MLXRandom.normal([1, size * size, 64], key: MLXRandom.key(7)) }
        let layout = try SequenceLayout(imageSlots: Array(repeating: false, count: states.dim(1)), conditions: [], target: (size, size))
        let cache = transformer.prefill(text: states, conditions: [], layout: layout)
        for t in [1000, 500, 100] as [Float] {
            outputs["transformer.v\(Int(t))"] = transformer.denoise(inputs[noiseKey]!, t: t, cache: cache, layout: layout).asType(.float32)
            eval(outputs["transformer.v\(Int(t))"]!)
        }
        Memory.clearCache()
    }
    if component == "all" || component == "vae" {
        try admit("vae \(tag)", expectedBytes: Int(4 * gb))
        let vae = try loadVAE(dtype)
        vae.decoderObserver = { stage, y in
            let f = y.asType(.float32)
            FileHandle.standardError.write(Data("vae \(stage): max |y| \(abs(f).max().item(Float.self)) finite \(MLX.all(MLX.isFinite(f)).item(Bool.self))\n".utf8))
        }
        if inputs["latent"] == nil { inputs["latent"] = MLXRandom.normal([1, 32, 32, 64], key: MLXRandom.key(11)) * 0.8 }
        outputs["vae.decoded"] = vae.decode(inputs["latent"]!)
        eval(outputs["vae.decoded"]!)
    }
    try save(arrays: inputs, url: inputsURL)
    try save(arrays: outputs, url: dir.appendingPathComponent("\(tag).safetensors"))
    if let reference = option("reference") {
        let ref = try loadArrays(url: dir.appendingPathComponent("\(reference).safetensors"))
        var rows: [[String: Any]] = []
        for key in outputs.keys.sorted() {
            guard let a = ref[key], let b = outputs[key], a.shape == b.shape else { rows.append(["key": key, "error": "missing or shape mismatch"]); continue }
            let a32 = a.asType(.float32), b32 = b.asType(.float32)
            let rel = (sqrt(sum(square(a32 - b32))) / sqrt(sum(square(a32)))).item(Float.self)
            let cos = (sum(a32 * b32) / (sqrt(sum(square(a32))) * sqrt(sum(square(b32))))).item(Float.self)
            let finite = MLX.all(MLX.isFinite(b32)).item(Bool.self)
            func n(_ v: Float) -> Any { v.isFinite ? v : String(describing: v) } // JSON has no NaN or infinity
            rows.append([
                "key": key, "relative_l2": n(rel), "cosine": n(cos), "finite": finite, "max_abs_reference": n(abs(a32).max().item(Float.self)),
                "max_abs_candidate": n(abs(b32).max().item(Float.self)), "elements": b.size, "pass_quant_policy": finite && rel <= 0.05 && cos >= 0.999
            ])
        }
        let data = try JSONSerialization.data(withJSONObject: ["reference": reference, "candidate": tag, "rows": rows], options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent("compare-\(tag)-vs-\(reference).json"))
        print(String(decoding: data, as: UTF8.self))
    }
}

/// Activation ranges of the text encoder in a half format: the largest |x| after every decoder layer.
func encoderRange() throws {
    try admit("encoder range", expectedBytes: Int(20 * gb))
    let encoder = try loadEncoder(dtype)
    let tokenize = try tokenizer()
    let prompt = option("prompt") ?? "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night"
    let tokens = PromptLayout(prompt: prompt, imageSlots: []).tokens(tokenize, imagePad: encoder.config.imageToken)
    let states = try encoder(tokens: tokens, pixels: nil, grids: [])
    print("tokens", tokens.count, "max |state|", abs(states.asType(.float32)).max().item(Float.self), "finite", MLX.all(MLX.isFinite(states)).item(Bool.self))
}

/// Per-step denoise time of two tiers on the same fixed prefix and latents, interleaved ABBA BAAB in one process so both
/// see the same machine load. Prints every window's seconds per step and each arm's median.
func stepBench() throws {
    Memory.memoryLimit = number("limit-gb", 60) * 1_000_000_000
    let arms = (option("arms") ?? "bf16,int8").split(separator: ",").map(String.init)
    try admit("step bench", expectedBytes: Int((Double(option("expect-gb") ?? "26") ?? 26) * gb))
    let inputs = try loadArrays(url: URL(fileURLWithPath: option("inputs") ?? "lab/qwen-image/components/inputs.safetensors"))
    guard let states = inputs["text.states"] else { throw LabError("Capture encoder states first") }
    let size = number("size", 1024) / 16, window = number("window", 3)
    let latents = MLXRandom.normal([1, size * size, 64], key: MLXRandom.key(9))
    let layout = try SequenceLayout(imageSlots: Array(repeating: false, count: states.dim(1)), conditions: [], target: (size, size))
    var models: [String: (QwenImageTransformer, PrefixCache)] = [:]
    for arm in arms {
        let t = try loadTransformer(arm)
        models[arm] = (t, t.prefill(text: states, conditions: [], layout: layout))
    }
    func run(_ arm: String) -> Double {
        let (t, cache) = models[arm]!
        var x = latents
        let start = Date()
        for i in 0..<window {
            // swiftlint:disable:next shorthand_operator
            x = x + Float(-0.02) * t.denoise(x, t: 1000 - Float(i) * 25, cache: cache, layout: layout).asType(.float32)
            eval(x)
        }
        return Date().timeIntervalSince(start) / Double(window)
    }
    for arm in arms { _ = run(arm) } // warm
    let order = arms.count == 2 ? [0, 1, 1, 0, 1, 0, 0, 1].map { arms[$0] } : arms + arms.reversed()
    var samples: [String: [Double]] = [:]
    for arm in order {
        let s = run(arm); samples[arm, default: []].append(s)
        FileHandle.standardError.write(Data(String(format: "%@ %.3f s/step\n", arm, s).utf8))
        try checkSwap(arm)
    }
    var report: [String: Any] = ["size": size * 16, "window_steps": window, "order": order, "date": ISO8601DateFormatter().string(from: Date())]
    for (arm, values) in samples { report[arm] = ["samples": values, "median": values.sorted()[values.count / 2]] }
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    if let path = option("report") { try data.write(to: URL(fileURLWithPath: path)) }
    print(String(decoding: data, as: UTF8.self))
}

/// Header-only inventory: every checkpoint tensor (names, shapes, dtypes from the local safetensors headers) against the
/// full-size module graph, through the same name map Load uses. No tensor payload is read.
func inventory() throws {
    var report: [String: Any] = [:]
    let components: [(String, Module, (String) -> String?, ([Int]) -> [Int])] = [
        ("text_encoder", Qwen3VLEncoder(try Qwen3VLConfig(json: json("text_encoder/config.json"))), QwenImageWeights.encoderName, { $0 }),
        ("transformer", QwenImageTransformer(try TransformerConfig(json: json("transformer/config.json"))), { QwenImageWeights.transformerName($0) }, { $0 }),
        ("vae", QwenImageVAE(try VAEConfig(json: json("vae/config.json"))), QwenImageWeights.vaeName, { $0 })
    ]
    for (directory, module, map, _) in components {
        let expected = Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { ($0.0, $0.1.shape) })
        var seen = Set<String>(), skipped: [String] = [], mismatched: [String] = [], dtypes: [String: Int] = [:], tensors = 0, values = 0
        for url in try archives(directory) {
            let archive = try TensorArchive(url)
            for (key, entry) in archive.entries {
                tensors += 1; dtypes[String(describing: entry.dtype), default: 0] += 1
                guard let name = map(key) else { skipped.append(key); continue }
                var shape = entry.shape
                if directory == "vae" {
                    if name.hasSuffix(".gamma") { shape = [shape.reduce(1, *)] } else if shape.count == 4 { shape = [shape[0], shape[2], shape[3], shape[1]] }
                }
                if expected[name] != shape { mismatched.append("\(key) \(entry.shape) → \(name) \(expected[name].map { "\($0)" } ?? "absent")") }
                seen.insert(name); values += entry.shape.reduce(1, *)
            }
        }
        let missing = Set(expected.keys).subtracting(seen).sorted()
        report[directory] = [
            "checkpoint_tensors": tensors, "loaded": seen.count, "skipped": skipped.sorted(), "module_parameters": expected.count, "missing": missing,
            "mismatched": mismatched, "dtypes": dtypes, "loaded_values": values
        ]
    }
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    if let path = option("report") { try data.write(to: URL(fileURLWithPath: path)) }
    print(String(decoding: data, as: UTF8.self))
}

do {
    switch mode {
    case "inventory": try inventory()
    case "step-bench": try stepBench()
    case "generate": try generate()
    case "components": try components()
    case "encoder-range": try encoderRange()
    case "tokens":
        let tokenize = try tokenizer()
        let layout = PromptLayout(prompt: option("prompt") ?? "a cat", imageSlots: (options["slots"] ?? []).compactMap(Int.init))
        print(layout.tokens(tokenize, imagePad: 151655).map(String.init).joined(separator: " "))
        print("drop", PromptLayout.dropIndex(tokenize))
    default:
        print("qwen-image-lab generate|components|encoder-range|tokens --weights <snapshot> --dtype bf16|fp16|int8|int4 …")
    }
} catch {
    FileHandle.standardError.write(Data("qwen-image-lab: \(error)\n".utf8)); exit(1)
}
