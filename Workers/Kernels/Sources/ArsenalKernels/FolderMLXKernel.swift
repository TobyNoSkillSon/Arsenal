// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXPreamble
import ArsenalKernelsCore

/// A folder kernel (`api = "mlx"`) run through `MLXFast.metalKernel`, with a compile preflight per instantiation.
///
/// MLX JIT-compiles a custom kernel inside `eval` and a compile error there is `fatalError` (it took Vireo's helper
/// down; skill "optimize-mlx-inference"). Before MLX sees a new instantiation (template values, input dtypes and
/// ranks, output dtypes, threadgroup size) this builds the exact source MLX will build (`MLXKernelSource` + MLX's own
/// preamble from the linked MLX), compiles it through Metal and creates the pipeline. Failure → the call returns nil
/// and the caller runs its stock expression; the reason is kept in `failure` for status. Verdicts persist by
/// (full source digest, threadgroup, OS build, GPU): a second load reuses them and MLX's compile is the only one.
public final class FolderMLXKernel: GatedKernel, @unchecked Sendable {
    public let loaded: LoadedKernel
    private let kernel: MLXFast.MLXFastKernel
    private let compiler: KernelCompiler
    private let verdicts: KernelVerdicts
    private let lock = NSLock()
    /// What decides MLX's generated source: per input dtype + how it is passed (scalar / constant / device), template
    /// values, output dtypes, threadgroup size. Integers only, so the per-call lookup costs no string building.
    private struct Instance: Hashable {
        var parts: [Int] = []
        init(inputs: [MLXArray], template: [(String, any KernelTemplateArg)], outputs: [DType], threadGroup: (Int, Int, Int)) {
            parts.reserveCapacity(inputs.count + 2 * template.count + outputs.count + 1)
            for x in inputs { parts.append(x.dtype.hashValue &* 4 &+ (x.ndim == 0 ? 1 : x.size < 8 ? 2 : 0)) }
            for (name, value) in template {
                parts.append(name.hashValue)
                switch value {
                case let v as Bool: parts.append(1); parts.append(v ? 1 : 0)
                case let v as Int: parts.append(2); parts.append(v)
                case let v as DType: parts.append(3); parts.append(v.hashValue)
                default: parts.append(4)
                }
            }
            for d in outputs { parts.append(d.hashValue) }
            parts.append(threadGroup.0 * threadGroup.1 * threadGroup.2)
        }
    }
    private var instances: [Instance: String?] = [:]
    /// MLX instantiation name + threadgroup → nil (passed) or the failure.
    private var checked: [String: String?] = [:]
    /// The first preflight failure (status line); nil while every instantiation compiled. Any failed preflight sets
    /// it, argument mismatches included: a gate then knows its "kernel on" run fell back to stock.
    public var failure: String? { lock.lock(); defer { lock.unlock() }; return firstFailure }
    private var firstFailure: String?
    public var gateName: String { loaded.spec.id }
    /// Calls that dispatched the kernel (engagement proof: a gate that never ran the kernel proves nothing).
    public var calls: Int { lock.lock(); defer { lock.unlock() }; return callCount }
    private var callCount = 0
    /// Instantiations compiled through Metal in this process (cache hits and persisted passes excluded).
    public private(set) var preflightCompiles = 0

    /// MLX's Metal preamble (`mlx::core::metal::utils()`), read from the linked MLX.
    public static let preamble = String(cString: arsenal_mlx_metal_utils())

    public init(_ loaded: LoadedKernel, compiler: KernelCompiler = .shared, verdicts: KernelVerdicts = .shared) throws {
        guard loaded.spec.api == .mlx else { throw KernelError.invalid("Kernel \(loaded.spec.id) is not an api = \"mlx\" kernel.") }
        self.loaded = loaded; self.compiler = compiler; self.verdicts = verdicts
        kernel = MLXFast.metalKernel(
            name: loaded.spec.entry, inputNames: loaded.spec.inputs, outputNames: loaded.spec.outputs,
            source: loaded.body, header: loaded.header)
    }

    /// Runs the kernel, or returns nil when this instantiation failed its preflight (caller: stock path).
    public func callAsFunction(
        _ inputs: [MLXArray], template: [(String, any KernelTemplateArg)] = [],
        grid: (Int, Int, Int), threadGroup: (Int, Int, Int),
        outputShapes: [[Int]], outputDTypes: [DType], initValue: Float? = nil
    ) -> [MLXArray]? {
        // Hot path: a cheap integer key per instantiation; the source-building preflight runs once per key.
        let key = Instance(inputs: inputs, template: template, outputs: outputDTypes, threadGroup: threadGroup)
        lock.lock()
        let known = instances[key]
        lock.unlock()
        let reason = known ?? preflight(inputs: inputs, template: template, outputDTypes: outputDTypes, threadGroup: threadGroup)
        lock.lock()
        if known == nil { instances[key] = .some(reason) }
        if reason == nil { callCount += 1 }
        lock.unlock()
        guard reason == nil else { return nil }
        return kernel(
            inputs, template: template.isEmpty ? nil : template, grid: grid, threadGroup: threadGroup,
            outputShapes: outputShapes, outputDTypes: outputDTypes, initValue: initValue)
    }

    /// nil when MLX can compile and dispatch this instantiation; else one sentence (also kept in `failure`). No GPU
    /// work: compile only.
    public func preflight(
        inputs: [MLXArray], template: [(String, any KernelTemplateArg)], outputDTypes: [DType],
        threadGroup: (Int, Int, Int)
    ) -> String? {
        let reason = check(inputs: inputs, template: template, outputDTypes: outputDTypes, threadGroup: threadGroup)
        if let reason {
            lock.lock()
            if firstFailure == nil { firstFailure = reason }
            lock.unlock()
        }
        return reason
    }

    private func check(
        inputs: [MLXArray], template: [(String, any KernelTemplateArg)], outputDTypes: [DType],
        threadGroup: (Int, Int, Int)
    ) -> String? {
        let spec = loaded.spec
        guard inputs.count == spec.inputs.count, outputDTypes.count == spec.outputs.count else {
            return "Kernel \(spec.id) called with \(inputs.count) inputs / \(outputDTypes.count) outputs; it declares \(spec.inputs.count) / \(spec.outputs.count)."
        }
        var described: [MLXKernelSource.Input] = []
        for input in inputs {
            guard let type = Self.metalType(input.dtype) else { return "Kernel \(spec.id): unsupported input dtype \(input.dtype)." }
            described.append(.init(type: type, ndim: input.ndim, size: input.size))
        }
        var outputTypes: [String] = []
        for dtype in outputDTypes {
            guard let type = Self.metalType(dtype) else { return "Kernel \(spec.id): unsupported output dtype \(dtype)." }
            outputTypes.append(type)
        }
        var args: [(String, MLXKernelSource.TemplateArg)] = []
        for (name, value) in template {
            switch value {
            case let v as Bool: args.append((name, .bool(v)))
            case let v as Int: args.append((name, .int(v)))
            case let v as DType:
                guard let type = Self.metalType(v) else { return "Kernel \(spec.id): unsupported template dtype \(v)." }
                args.append((name, .type(type)))
            default: return "Kernel \(spec.id): unsupported template argument \(name)."
            }
        }
        let threads = threadGroup.0 * threadGroup.1 * threadGroup.2
        let name = MLXKernelSource.kernelName(name: spec.entry, inputs: described, outputTypes: outputTypes, template: args)
        let memo = name + "|tg\(threads)"
        lock.lock()
        if let known = checked[memo] { lock.unlock(); return known }
        lock.unlock()

        let generated = MLXKernelSource.generate(
            name: spec.entry, header: loaded.header, body: loaded.body, inputNames: spec.inputs,
            inputs: described, outputNames: spec.outputs, outputTypes: outputTypes, template: args)
        let source = Self.preamble + generated.source
        let key = KernelVerdicts.key("mlx-compile", KernelDigest.sha256(source), "tg\(threads)")
        let result = verdicts.verdict(key) {
            lock.lock(); preflightCompiles += 1; lock.unlock()
            let reason = compiler.check(source: source, function: generated.kernelName, threadsPerThreadgroup: threads)
            return .init(pass: reason == nil, detail: reason ?? "compiled")
        }
        let reason: String? = result.verdict.pass ? nil : "Kernel \(spec.id): \(result.verdict.detail)"
        lock.lock()
        checked[memo] = .some(reason)
        lock.unlock()
        return reason
    }

    /// MLX's `get_type_string`.
    static func metalType(_ dtype: DType) -> String? {
        switch dtype {
        case .float32: return "float"
        case .float16: return "float16_t"
        case .bfloat16: return "bfloat16_t"
        case .float64: return "double"
        case .complex64: return "complex64_t"
        case .bool: return "bool"
        case .int8: return "int8_t"
        case .int16: return "int16_t"
        case .int32: return "int32_t"
        case .int64: return "int64_t"
        case .uint8: return "uint8_t"
        case .uint16: return "uint16_t"
        case .uint32: return "uint32_t"
        case .uint64: return "uint64_t"
        @unknown default: return nil
        }
    }
}

public extension ExactComparison {
    /// Bitwise comparison of two evaluated arrays (shape, dtype, then every byte).
    static func compare(_ a: MLXArray, _ b: MLXArray) -> ExactComparison {
        guard a.shape == b.shape else { return .layout("shapes \(a.shape) vs \(b.shape)") }
        guard a.dtype == b.dtype else { return .layout("dtypes \(a.dtype) vs \(b.dtype)") }
        let element: Element
        switch a.dtype {
        case .float16: element = .float16
        case .bfloat16: element = .bfloat16
        case .float32: element = .float32
        default: element = .bytes
        }
        return compare(a.asData(access: .copy).data, b.asData(access: .copy).data, element: element)
    }
}
