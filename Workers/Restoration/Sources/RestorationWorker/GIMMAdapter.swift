// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import GIMMNative
import MLX
import ArsenalWire

/// GIMM-VFI-R: frame interpolation at arbitrary timesteps through the shared `interpolate_video` contract. RAFT and the
/// pair features run once per frame pair; each requested t then runs the motion model and the decoders.
final class GIMMAdapter: NativeModel {
    let architecture = "gimm_vfi_r"
    let operations: Set<String> = ["interpolate_video"]
    private var model: GIMMVFIR?
    private var weights: [String: MLXArray] = [:]
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { weights }
    /// Convolutions, the hyponet's weight-normalised matrices and vectors: no tier quantizes them; the model ships 16-bit.
    static func eligible(_ name: String, _ shape: [Int]) -> Bool { false }
    static let checkpoint = "gimmvfi_r_arb.pt"

    /// Get: read the maker's torch checkpoint without executing it, keep its tensor names, drop BatchNorm step counters
    /// and RAFT's `norm3` aliases (checked equal to `downsample.1`), transpose convolution kernels to OHWI and store FP16.
    static let conversion = NativeConversion(recipe: "cast:restoration-gimm-vfi-r-v1") { context, cancellation in
        try cancellation.check()
        // Load runs FP16 only, so Get makes nothing else (a BF16 snapshot could never load).
        guard context.precision == "fp16" else { throw MediaBridge.error("GIMM-VFI-R ships at FP16 only.") }
        let source = try TorchCheckpoint(context.source.appendingPathComponent(checkpoint))
        let tensors = try source.tensors(at: ["state_dict"])
        var out: [String: MLXArray] = [:]
        for (name, tensor) in tensors.sorted(by: { $0.key < $1.key }) {
            try cancellation.check()
            if name.hasSuffix(".num_batches_tracked") { continue }
            if name.contains(".norm3.") {
                let alias = name.replacingOccurrences(of: ".norm3.", with: ".downsample.1.")
                guard let twin = tensors[alias], MLX.arrayEqual(try source.array(tensor), try source.array(twin), stream: .cpu).item(Bool.self) else {
                    throw WorkerFailure(code: "precision", message: "GIMM-VFI-R checkpoint: \(name) differs from \(alias)")
                }
                continue
            }
            var array = try source.array(tensor)
            guard array.dtype == .float32 else { throw WorkerFailure(code: "precision", message: "GIMM-VFI-R expects the float32 checkpoint; \(name) is \(array.dtype)") }
            if array.ndim == 4 { array = array.transposed(0, 2, 3, 1) }
            let half = array.asType(context.half, stream: .cpu)
            guard !MLX.any(.!MLX.isFinite(half, stream: .cpu), stream: .cpu).item(Bool.self) else {
                throw WorkerFailure(code: "precision", message: "GIMM-VFI-R tensor overflows \(context.precision): \(name)")
            }
            out[name] = half
        }
        try GIMMWeights.validate(out, half: context.half)
        try cancellation.check()
        try MLX.save(arrays: out, url: context.destination.appendingPathComponent("model.safetensors"))
        return try context.receipt(cancellation: cancellation)
    }

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        guard context.precision == "fp16" else { throw MediaBridge.error("GIMM-VFI-R runs at FP16 only.") }
        let w = context.randomWeights ? GIMMWeights.random(dtype: .float16) : try MLX.loadArrays(url: context.weights.appendingPathComponent("model.safetensors"))
        try GIMMWeights.validate(w, half: .float16)
        eval(Array(w.values))
        model = GIMMVFIR(weights: w); weights = w
    }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "interpolate_video", let model else { throw MediaBridge.error("GIMM-VFI-R is not loaded or the operation is unsupported.") }
        return try InterpolationDriver.run(input, cancellation: cancellation, emit: emit) { a, b in
            try cancellation.check()
            guard a.dim(1) >= 16, a.dim(2) >= 16 else { throw MediaBridge.error("GIMM-VFI-R needs frames of at least 16 pixels on each side.") }
            let padding = GIMMPadding(height: a.dim(1), width: a.dim(2))
            let pair = try model.pair(padding.pad(a), padding.pad(b), scale: padding.scale) { try cancellation.check() }
            return { t in
                try cancellation.check()
                let frame = padding.unpad(model.frame(pair, t: Float(t)))
                eval(frame)
                return frame
            }
        }
    }
    func unload() { model = nil; weights = [:] }
}
