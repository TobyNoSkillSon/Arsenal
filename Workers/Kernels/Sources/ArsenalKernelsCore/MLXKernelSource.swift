// SPDX-License-Identifier: MIT
import Foundation

/// The exact source MLX builds for an `MLXFast.metalKernel` instantiation, so it can be compiled through Metal first
/// (where a compile error is an error) instead of inside MLX (where it is `fatalError`).
///
/// Port of mlx `backend/common/metal_kernel.cpp` (`write_signature`, `write_template`, `make_template_hash` and the
/// kernel-name rules of `metal_kernel`) at mlx-swift 901941965d82e4a216d4d117231d847d194c563d (MLX 0.32). MLX then
/// compiles `metal::utils() + source` with MathMode::Safe and Metal 4.0 (macOS 26) / 4.1 (macOS 27). A different MLX
/// pin needs this file re-checked against that source (`MLXKernelSourceTests` holds a golden copy).
public enum MLXKernelSource {
    /// How MLX will pass one input: its Metal type, rank and element count decide the generated parameter.
    public struct Input: Sendable, Equatable {
        public var type: String
        public var ndim: Int
        public var size: Int
        public init(type: String, ndim: Int, size: Int) { self.type = type; self.ndim = ndim; self.size = size }
    }
    public enum TemplateArg: Sendable, Equatable {
        case int(Int), bool(Bool), type(String)
    }

    /// MLX's `get_type_string` for the dtype names `DType` prints.
    public static func metalType(_ dtype: String) -> String? {
        switch dtype {
        case "float32": return "float"
        case "float16": return "float16_t"
        case "bfloat16": return "bfloat16_t"
        case "float64": return "double"
        case "complex64": return "complex64_t"
        case "bool": return "bool"
        case "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64": return dtype + "_t"
        default: return nil
        }
    }

    /// MLX passes inputs with fewer elements in the constant address space.
    static let maxConstantArraySize = 8

    /// Attributes MLX adds to the signature when their name occurs in the body, in MLX's order.
    static let attributes: [(String, String)] = [
        ("dispatch_quadgroups_per_threadgroup", "uint"), ("dispatch_simdgroups_per_threadgroup", "uint"),
        ("dispatch_threads_per_threadgroup", "uint3"), ("grid_origin", "uint3"), ("grid_size", "uint3"),
        ("quadgroup_index_in_threadgroup", "uint"), ("quadgroups_per_threadgroup", "uint"),
        ("simdgroup_index_in_threadgroup", "uint"), ("simdgroups_per_threadgroup", "uint"),
        ("thread_execution_width", "uint"), ("thread_index_in_quadgroup", "uint"),
        ("thread_index_in_simdgroup", "uint"), ("thread_index_in_threadgroup", "uint"),
        ("thread_position_in_grid", "uint3"), ("thread_position_in_threadgroup", "uint3"),
        ("threadgroup_position_in_grid", "uint3"), ("threadgroups_per_grid", "uint3"), ("threads_per_grid", "uint3"),
        ("threads_per_simdgroup", "uint"), ("threads_per_threadgroup", "uint3")
    ]

    static func templateDefinition(_ template: [(String, TemplateArg)]) -> String {
        "<"
            + template.map { _, arg in
                switch arg {
                case .int(let v): return String(v)
                case .bool(let v): return v ? "1" : "0" // std::ostream without boolalpha
                case .type(let t): return t
                }
            }.joined(separator: ", ") + ">"
    }

    static func templateHash(_ definition: String) -> String {
        var out = ""
        var chars = Array(definition)[...]
        while let c = chars.popFirst() {
            if c == "<" || c == ">" { out.append("_") } else if c == ",", chars.first == " " { out.append("_"); chars = chars.dropFirst() } else { out.append(c) }
        }
        return String(out.dropLast())
    }

    /// MLX's host name for this instantiation (also the key of MLX's in-process library cache, with the source hash).
    public static func kernelName(name: String, inputs: [Input], outputTypes: [String], template: [(String, TemplateArg)]) -> String {
        var kernelName = "custom_kernel_" + name
        if !template.isEmpty { kernelName += "_" + templateHash(templateDefinition(template)) }
        for input in inputs {
            kernelName += "_" + input.type
            if input.ndim == 0 { kernelName += "s" } else if input.size < maxConstantArraySize { kernelName += "c" }
        }
        for type in outputTypes { kernelName += "_" + type }
        return kernelName
    }

    /// The kernel source MLX generates (without the preamble) and its host name.
    public static func generate(
        name: String, header: String, body: String, inputNames: [String], inputs: [Input],
        outputNames: [String], outputTypes: [String], template: [(String, TemplateArg)],
        atomicOutputs: Bool = false
    ) -> (source: String, kernelName: String) {
        let kernelName = kernelName(name: name, inputs: inputs, outputTypes: outputTypes, template: template)
        let shapeInfos = inputNames.map { n in (body.contains(n + "_shape"), body.contains(n + "_strides"), body.contains(n + "_ndim")) }
        let attrs = attributes.filter { body.contains($0.0) }.map { "  \($0.1) \($0.0) [[\($0.0)]]" }

        var s = header
        if !template.isEmpty {
            s +=
                "template <"
                + template.map { name, arg in
                    switch arg {
                    case .int: return "int " + name
                    case .bool: return "bool " + name
                    case .type: return "typename " + name
                    }
                }.joined(separator: ", ") + ">\n"
        }
        s += "[[kernel]] void " + kernelName + "(\n"
        var index = 0
        for (i, input) in inputs.enumerated() {
            let location = input.size < maxConstantArraySize ? "constant" : "device"
            let ref = input.ndim == 0 ? "&" : "*"
            s += "  const \(location) \(input.type)\(ref) \(inputNames[i]) [[buffer(\(index))]],\n"
            index += 1
            if input.ndim > 0 {
                if shapeInfos[i].0 { s += "  const constant int* \(inputNames[i])_shape [[buffer(\(index))]],\n"; index += 1 }
                if shapeInfos[i].1 { s += "  const constant int64_t* \(inputNames[i])_strides [[buffer(\(index))]],\n"; index += 1 }
                if shapeInfos[i].2 { s += "  const constant int& \(inputNames[i])_ndim [[buffer(\(index))]],\n"; index += 1 }
            }
        }
        for (i, name) in outputNames.enumerated() {
            let type = atomicOutputs ? "atomic<\(outputTypes[i])>" : outputTypes[i]
            s += "  device \(type)* \(name) [[buffer(\(index))]]"
            // MLX's own condition (index counts shape buffers too); reproduced as is, quirks included.
            s += index < inputs.count + outputNames.count - 1 || !attrs.isEmpty ? ",\n" : ") {\n"
            index += 1
        }
        for (i, attr) in attrs.enumerated() { s += attr + (i < attrs.count - 1 ? ",\n" : ") {\n") }
        s += body + "\n}\n"
        if !template.isEmpty {
            let definition = kernelName + templateDefinition(template)
            s += "\ntemplate [[host_name(\"\(kernelName)\")]] [[kernel]] decltype(\(definition)) \(definition);\n"
        }
        return (s, kernelName)
    }
}
