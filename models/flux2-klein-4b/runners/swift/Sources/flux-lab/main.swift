// SPDX-License-Identifier: MIT
// GPU component gates and short precision checks. Never installed or shipped.
import Darwin
import Flux2Klein
import Foundation
import MLX
import MLXRandom
import MLXNN
import MLXFast

func fail(_ message: String) -> Never { FileHandle.standardError.write(Data((message + "\n").utf8)); exit(1) }
var args = Array(CommandLine.arguments.dropFirst())
@MainActor func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let value = args[i + 1]; args.removeSubrange(i...(i + 1)); return value
}
func dtype(_ name: String?) -> DType {
    switch name {
    case "fp16": .float16
    case nil, "bf16": .bfloat16
    default: fail("only fp16 and bf16 are permitted")
    }
}
let type = dtype(option("--dtype"))
let precision = KleinPrecision(weights: type, encoder: type, activations: type, vae: type)
let quantBits = option("--quant").flatMap(Int.init)
if let quantBits, ![4, 8].contains(quantBits) { fail("quant must be 4 or 8") }
let count = Int(option("--runs") ?? "3")!
Memory.cacheLimit = 256 << 20
guard args.count == 4 else { fail("flux-lab te|dit|vae|steps <snapshot> <reference-dir> <output-dir> [--dtype fp16|bf16] [--quant 8|4]") }
let snapshot = URL(fileURLWithPath: args[1]), dir = URL(fileURLWithPath: args[2]), out = URL(fileURLWithPath: args[3])
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
guard let spec = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("case.json"))) as? [String: Any],
    let prompt = spec["prompt"] as? String
else { fail("malformed case.json") }
let sigmas = (spec["sigmas"] as? [Double]) ?? KleinSchedule.sigmas(seqLen: 4096, steps: 4).map(Double.init)
func read(_ name: String) throws -> MLXArray { try loadArrays(url: dir.appendingPathComponent(name + ".safetensors"))["value"]!.asType(type) }
func saved(_ name: String) throws -> MLXArray { try loadArrays(url: out.appendingPathComponent(name + ".safetensors"))["value"]!.asType(type) }
func dump(_ name: String, _ array: MLXArray) throws {
    eval(array)
    guard array.dtype == type else { fail("dtype drift: \(name) \(array.dtype)") }
    try save(arrays: ["value": array], url: out.appendingPathComponent(name + ".safetensors"))
}
func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var n = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(n)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &n) }
    }
    return result == KERN_SUCCESS ? Double(info.ledger_phys_footprint_peak) / 1e6 : 0
}
func emit(_ object: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self)); fflush(stdout)
}
var meta: [String: Any] = ["dtype": String(describing: type), "quant_bits": quantBits ?? 16, "machine_in_use": true]
switch args[0] {
case "images":
    // One unmeasured warm-up per resident arm, then newline commands permit a controller to alternate arms.
    let pipeline = try KleinPipeline.load(snapshot: snapshot, precision: precision)
    func image(_ label: String) throws -> [String: Any] {
        let start = Date()
        let result = try pipeline.generate(prompt: prompt, width: 1024, height: 1024, steps: 4, seed: 24301)
        let path = out.appendingPathComponent(label + ".png")
        try PNG.write(result.pixels, width: result.width, height: result.height, to: path)
        return [
            "label": label, "seconds_per_image": Date().timeIntervalSince(start), "steps_s": result.timings.steps,
            "footprint_peak_mb": footprintMB(), "dtype": String(describing: type), "output": path.path
        ]
    }
    try emit(["event": "ready", "warmup": image("warmup")])
    for i in 0..<count {
        guard readLine() != nil else { fail("missing interleaved image command") }
        try emit(image("image-\(i + 1)"))
    }
case "ops":
    let x = MLXArray([-6.84375, -4, -1, -0.1, 0, 0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16].map(Float16.init), [2, 8]).asType(type)
    try dump("ops-silu", silu(x))
    try dump("ops-rms", x * rsqrt(mean(square(x), axis: -1, keepDims: true) + 1e-6))
    let large = MLXArray([Float16(700), -700, 300, -300], [1, 4])
    try dump("ops-rms-large", MLXFast.rmsNorm(large, weight: MLXArray.ones([4], dtype: .float16), eps: 1e-6).asType(type))
    try dump("ops-rope", MLX.cos(MLXArray(Array(stride(from: 0, to: 16, by: 2))).asType(type) / 16) * x)
    try dump("ops-time", MLX.cos(MLXArray(Float16(1000)).asType(type) * MLX.exp(-Float(Foundation.log(10000.0)) * MLXArray(0..<128).asType(type) / 128)))
case "te":
    let tokenizer = try KleinTextEncoder.tokenizer(directory: snapshot.appendingPathComponent("tokenizer"))
    let encoder = KleinTextEncoder(encoder: try KleinWeights.textEncoder(snapshot: snapshot, dtype: type), tokenizer: tokenizer)
    let (ids, valid) = encoder.tokenize(prompt)
    meta["ids"] = ids; meta["valid"] = valid
    try dump("short-embeds", encoder.encode(prompt, activations: type, length: 32))
    try dump("embeds", encoder.encode(prompt, activations: type))
case "dit", "steps", "diagnose":
    let dit = try KleinWeights.transformer(snapshot: snapshot, dtype: type)
    if let quantBits { KleinWeights.quantize(dit, bits: quantBits); Memory.clearCache() }
    let (cos, sin) = KleinPipeline.ropeTables(textLength: 512, gridH: 64, gridW: 64, dtype: type)
    let e = try saved("embeds")[.newAxis]
    func velocity(_ x: MLXArray, _ i: Int) -> MLXArray {
        dit(x, encoder: e, timestep: MLXArray([Float16(sigmas[i] * 1000)]).asType(type), cos: cos, sin: sin, activations: type)
    }
    if args[0] == "diagnose" {
        let x = try read("latents-0")
        let result = dit(x, encoder: e, timestep: MLXArray([Float16(1000)]).asType(type), cos: cos, sin: sin, activations: type) { name, value in
            eval(value)
            print("\(name): max \(abs(value).max().item(Float.self)), finite \(all(isFinite(value)).item(Bool.self))", terminator: "\n")
            fflush(stdout)
        }
        eval(result)
    } else if args[0] == "steps" {
        let x = try read("latents-0")
        var times: [Double] = []
        for i in 0...count {
            let t = Date()
            let y = velocity(x, 1); eval(y)
            if i > 0 { times.append(Date().timeIntervalSince(t)) }
        }
        meta["steps_s"] = times
    } else {
        try dump(
            "tiny-velocity",
            dit(
                try read("tiny-x"), encoder: try read("tiny-e"), timestep: MLXArray([Float16(1000)]).asType(type),
                cos: try read("block-cos"), sin: try read("block-sin"), activations: type))
        try dump(
            "block-out-x",
            dit.componentBlock(
                try read("block-x"), encoder: try read("block-e"),
                img: try (0..<2).map { i in (try read("block-img-\(i)-0"), try read("block-img-\(i)-1"), try read("block-img-\(i)-2")) },
                txt: try (0..<2).map { i in (try read("block-txt-\(i)-0"), try read("block-txt-\(i)-1"), try read("block-txt-\(i)-2")) },
                cos: try read("block-cos"), sin: try read("block-sin")
            ).0)
        var x = try read("latents-0")
        var times: [Double] = []
        for i in 0..<4 {
            let t = Date()
            let v = velocity(x, i)
            x = x + MLXArray(Float16(sigmas[i + 1] - sigmas[i])).asType(type) * v // swiftlint:disable:this shorthand_operator
            eval(x); times.append(Date().timeIntervalSince(t))
        }
        try dump("latents", x); meta["steps_s"] = times
    }
case "vae":
    let vae = try KleinWeights.vae(snapshot: snapshot, dtype: type)
    func decode(_ packed: MLXArray, _ grid: Int) -> MLXArray { vae.decode(vae.latents(packed, gridH: grid, gridW: grid), dtype: type) }
    try dump("vae-small", decode(try read("vae-small-input"), 4))
    let image = decode(try saved("latents"), 64)
    try dump("image", image)
    try PNG.write(KleinPipeline.pixels(image), width: 1024, height: 1024, to: out.appendingPathComponent("image.png"))
default: fail("unknown command \(args[0])")
}
meta["mlx_peak_mb"] = Double(Memory.peakMemory) / 1e6; meta["footprint_peak_mb"] = footprintMB()
try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]).write(to: out.appendingPathComponent(args[0] + ".json"))
print("\(args[0]) \(type): MLX peak \(Int(Memory.peakMemory / 1_000_000)) MB, footprint \(Int(footprintMB())) MB")
