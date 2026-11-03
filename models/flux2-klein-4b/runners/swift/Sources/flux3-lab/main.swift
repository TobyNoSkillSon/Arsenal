// SPDX-License-Identifier: MIT
// FLUX 3 Image Klein lab entry point: tokenizer parity, real-weight component checks on fixed tensors, and single
// generations, against captures from mflux (lab/flux3/scripts). Never installed or shipped.
//   flux3-lab tokenize <text_encoder dir> <prompts.json> <out.json>
//   flux3-lab components <snapshot> <inputs.safetensors> <out.safetensors>
//   flux3-lab generate <snapshot> <out.png> --prompt P [--seed S] [--width W] [--height H] [--steps N] [--image ref]... [--latents out.safetensors]
import Darwin
import Flux2Klein
import Flux3Klein
import Foundation
import MLX
import MLXNN

func fail(_ message: String) -> Never { FileHandle.standardError.write(Data((message + "\n").utf8)); exit(1) }
var args = Array(CommandLine.arguments.dropFirst())
@MainActor func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let value = args[i + 1]; args.removeSubrange(i...(i + 1)); return value
}
@MainActor func options(_ name: String) -> [String] { var out: [String] = []; while let v = option(name) { out.append(v) }; return out }
func emit(_ object: [String: Any]) {
    let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
    print(String(decoding: data, as: UTF8.self)); fflush(stdout)
}
func peakGB() -> Double {
    var info = task_vm_info_data_t()
    var n = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(n)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &n) }
    }
    return result == KERN_SUCCESS ? Double(info.ledger_phys_footprint_peak) / 1e9 : 0
}

if let metallib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { GPU.metallib = URL(fileURLWithPath: metallib) }
Memory.cacheLimit = 256 << 20
guard let command = args.first else { fail("flux3-lab tokenize|components|generate …") }
args.removeFirst()

switch command {
case "tokenize":
    guard args.count == 3 else { fail("flux3-lab tokenize <text_encoder dir> <prompts.json> <out.json>") }
    let tokenizer = try Flux3Prompt.tokenizer(directory: URL(fileURLWithPath: args[0]))
    guard let prompts = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[1]))) as? [String] else { fail("prompts.json must be a list of strings") }
    let out = prompts.map { prompt -> [String: Any] in
        let (ids, valid) = Flux3Prompt.tokenize(prompt, tokenizer: tokenizer)
        return ["ids": ids, "valid": valid]
    }
    try JSONSerialization.data(withJSONObject: out).write(to: URL(fileURLWithPath: args[2]))
    emit(["prompts": prompts.count])

case "components":
    guard args.count == 3 else { fail("flux3-lab components <snapshot> <inputs.safetensors> <out.safetensors>") }
    let snapshot = URL(fileURLWithPath: args[0])
    let input = try loadArrays(url: URL(fileURLWithPath: args[1]))
    func need(_ name: String) -> MLXArray {
        guard let value = input[name] else { fail("inputs have no \(name)") }
        return value
    }
    var out: [String: MLXArray] = [:]
    var start = Date()
    // Text encoder: one component at a time, released before the next loads.
    do {
        let text = try Flux3Weights.textEncoder(directory: snapshot.appendingPathComponent("text_encoder"))
        let ids = need("te_ids"), valid = need("te_valid").item(Int.self)
        out["te_embeds"] = text(ids, valid: valid); eval(out["te_embeds"]!)
    }
    emit(["component": "text_encoder", "seconds": Date().timeIntervalSince(start), "peak_gb": peakGB()])
    Memory.clearCache(); start = Date()
    do {
        let t = try Flux3Weights.transformer(directory: snapshot.appendingPathComponent("transformer"))
        let gh = need("grid").asArray(Int32.self)
        let textLen = need("te_embeds_ref").dim(1)
        func rope(_ cond: [[Float]]) -> Flux3RopeTables {
            Flux3RopeTables(
                image: Flux3Positions.imageIDs(gridH: Int(gh[0]), gridW: Int(gh[1])), cond: cond, text: Flux3Prompt.textIDs(textLen), config: t.config, dtype: .bfloat16)
        }
        let condIDs = need("cond_ids").asArray(Float.self)
        let ids = stride(from: 0, to: condIDs.count, by: 4).map { Array(condIDs[$0..<($0 + 4)]) }
        out["velocity_t2i"] = t(need("latents"), text: need("te_embeds_ref"), cond: nil, t: need("t"), next: need("t_next"), rope: rope([])) { s in
            out["mode0_image"] = s.modes[0]["image"]; out["single0"] = s.singles[0]; out["single_last"] = s.singles[s.singles.count - 1]
        }
        out["velocity_edit"] = t(need("latents"), text: need("te_embeds_ref"), cond: need("cond_tokens"), t: need("t"), next: need("t_next"), rope: rope(ids))
        eval(out.values.map { $0 })
    }
    emit(["component": "transformer", "seconds": Date().timeIntervalSince(start), "peak_gb": peakGB()])
    Memory.clearCache(); start = Date()
    do {
        let (encoder, decoder) = try Flux3Weights.vae(directory: snapshot.appendingPathComponent("vae"))
        out["vae_tokens"] = encoder.tokens(need("vae_image").transposed(0, 2, 3, 1), statistics: decoder.statistics)
        let g = need("vae_grid").asArray(Int32.self)
        out["vae_decoded"] = decoder.decode(decoder.latents(state: need("vae_decode_in"), gridH: Int(g[0]), gridW: Int(g[1]), dtype: .bfloat16), dtype: .bfloat16)
            .transposed(0, 3, 1, 2)
        eval(out["vae_tokens"]!, out["vae_decoded"]!)
    }
    emit(["component": "vae", "seconds": Date().timeIntervalSince(start), "peak_gb": peakGB()])
    try save(arrays: out, url: URL(fileURLWithPath: args[2]))

case "generate":
    guard let prompt = option("--prompt") else { fail("--prompt is required") }
    var request = Flux3Request(prompt: prompt)
    guard let seed = UInt64(option("--seed") ?? "0"), let steps = Int(option("--steps") ?? "16") else { fail("--seed and --steps take integers") }
    request.seed = seed; request.steps = steps
    request.width = option("--width").flatMap(Int.init); request.height = option("--height").flatMap(Int.init)
    request.references = try options("--image").map { try RGBImage.read(URL(fileURLWithPath: $0)) }
    let latentsOut = option("--latents")
    guard args.count == 2 else { fail("flux3-lab generate <snapshot> <out.png> --prompt …") }
    if request.references.isEmpty { request.width = request.width ?? 1024; request.height = request.height ?? 1024 }
    var start = Date()
    let pipeline = try Flux3Pipeline.load(snapshot: URL(fileURLWithPath: args[0]))
    emit(["phase": "load", "seconds": Date().timeIntervalSince(start), "peak_gb": peakGB()])
    start = Date()
    let out = try pipeline.generate(request) { step, steps in if step == steps { emit(["phase": "denoised", "seconds": Date().timeIntervalSince(start)]) } }
    try PNG.write(out.pixels, width: out.width, height: out.height, to: URL(fileURLWithPath: args[1]))
    if let latentsOut { try save(arrays: ["latents": out.latents], url: URL(fileURLWithPath: latentsOut)) }
    emit(["phase": "done", "seconds": Date().timeIntervalSince(start), "width": out.width, "height": out.height, "peak_gb": peakGB()])

default:
    fail("unknown command \(command)")
}
