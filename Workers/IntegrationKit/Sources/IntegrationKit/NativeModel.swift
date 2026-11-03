// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import MLX
import MLXNN
import MLXRandom
import ModelPrecision

public struct NativeLoadContext {
    public let weights: URL
    public let folder: URL
    public var architecture: String? { config["architecture"]?.string }
    public let precision: String
    public let config: JSONObject
    public let assets: [String: URL]
    public let randomWeights: Bool
    public init(_ request: JSONObject) throws {
        guard let weights = request["weights_dir"]?.string, let folder = request["model_dir"]?.string,
            let precision = request["dtype"]?.string, ["fp16", "bf16", "int8", "int4"].contains(precision)
        else {
            throw WorkerFailure(code: "invalid_request", message: "Load requires model_dir, weights_dir and fp16/bf16/int8/int4 dtype.")
        }
        self.weights = URL(fileURLWithPath: weights); self.folder = URL(fileURLWithPath: folder); self.precision = precision
        config = request["config"]?.object ?? [:]
        assets = (request["assets"]?.object ?? config["assets"]?.object ?? [:]).compactMapValues { $0.string.map { URL(fileURLWithPath: $0) } }
        randomWeights = config["random_weights"]?.bool == true
        if randomWeights, ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] != "1" {
            throw WorkerFailure(code: "invalid_request", message: "Random weights require an isolated ARSENAL_NATIVE_LAB worker.")
        }
    }
}

public final class NativeCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var canceled = false
    public init() {}
    public func cancel() { lock.withLock { canceled = true } }
    public func check() throws { if lock.withLock({ canceled }) { throw WorkerFailure(code: "cancelled", message: "Request cancelled.") } }
}

/// One model owns its tensors and operation implementations. Calls are serialized by NativeFamilyWorker.
/// The app validates input/output schemas; adapters still validate engine-specific shape and dtype requirements.
public protocol NativeModel: AnyObject {
    var architecture: String { get }
    var operations: Set<String> { get }
    var peakBytes: Int { get }
    var parameters: [String: MLXArray] { get }
    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws
    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject
    func unload()
}

/// Adapters that turn generated text into a structured result expose that same parser to the lab checker, so
/// maker-documented example outputs (`gate/parser-fixtures.json`) pass through the serving code path and the production
/// output validator. `run` must call this parser on its generated text; a reply the parser cannot accept throws
/// `WorkerFailure(code: NativeOutputParsingFailure.code, ...)`, a code reserved for exactly that rejection.
public protocol NativeOutputParsing {
    /// Parse `raw` model output for `operation` as `run` does after generation. `input` is the request (with `$INBOX`
    /// files already resolved), for parsers that need the source image size, template or language.
    func parseOutput(_ operation: String, input: JSONObject, raw: String) throws -> JSONObject
}

public enum NativeOutputParsingFailure {
    /// The one error code for generated output the operation's parser rejects (random weights produce exactly this).
    public static let code = "invalid_model_output"
}

public struct NativeRegistration {
    public let modelTypes: Set<String>
    public let make: () -> any NativeModel
    public let conversion: NativeConversion?
    public init(_ modelTypes: Set<String>, conversion: NativeConversion? = nil, make: @escaping () -> any NativeModel) {
        self.modelTypes = modelTypes; self.conversion = conversion; self.make = make
    }
}

extension NativeRegistration {
    /// The verified folder selector is authoritative; checkpoint model_type is the legacy fallback.
    public static func resolve(checkpoint: JSONObject, architecture: String?, registrations: [NativeRegistration]) throws -> NativeRegistration {
        let checkpointType = checkpoint["model_type"]?.string
        guard let type = architecture ?? checkpointType, !type.isEmpty else {
            throw WorkerFailure(code: "unsupported_model", message: "Checkpoint needs model_type or a verified folder architecture.")
        }
        let matches = registrations.filter { $0.modelTypes.contains(type) }
        guard matches.count == 1 else { throw WorkerFailure(code: "unsupported_model", message: "Architecture must match exactly one registered adapter: " + type) }
        return matches[0]
    }
}

public enum NativeParameters {
    public static func inventory(_ parameters: [String: MLXArray]) -> JSONObject {
        parameters.mapValues { tensor in ["shape": .array(tensor.shape.map { .number(Double($0)) }), "dtype": .string(String(describing: tensor.dtype))] }
    }
    public static func requireHalf(_ parameters: [String: MLXArray]) throws {
        let bad = parameters.filter { [.float32, .float64].contains($0.value.dtype) }.keys.sorted()
        guard bad.isEmpty else {
            throw WorkerFailure(code: "precision", message: "Learned weights must be half or packed; found full precision: " + bad.prefix(8).joined(separator: ", "))
        }
    }
    public static func initializeRandom(_ module: Module, half: DType = .float16, seed: UInt64 = 0) throws {
        guard half == .float16 || half == .bfloat16 else { throw PrecisionError("Random model weights must be half") }
        MLXRandom.seed(seed)
        var updates: [String: MLXArray] = [:]
        for (key, value) in module.parameters().flattened() where [.float16, .bfloat16, .float32, .float64].contains(value.dtype) {
            // Retain initialization semantics (norm scales and BN variance), converting each parameter before execution.
            updates[key] = value.asType(half)
        }
        try module.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
        try requireHalf(Dictionary(uniqueKeysWithValues: module.parameters().flattened()))
    }
}
