// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN

public struct PrecisionError: LocalizedError {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// An indexed file reader: only the requested tensor is copied, never a complete dense checkpoint.
public final class TensorArchive {
    public struct Entry: Sendable {
        public let dtype: DType
        public let shape: [Int]
        public let offset: UInt64
        public let bytes: Int
    }
    public let entries: [String: Entry]
    private let file: FileHandle
    public static let types: [String: DType] = [
        "F16": .float16, "BF16": .bfloat16, "F32": .float32, "F64": .float64,
        "I8": .int8, "U8": .uint8, "I16": .int16, "U16": .uint16, "I32": .int32, "U32": .uint32,
        "I64": .int64, "U64": .uint64, "BOOL": .bool
    ]
    public init(_ url: URL) throws {
        let f = try FileHandle(forReadingFrom: url)
        do {
            let size = try f.seekToEnd(); try f.seek(toOffset: 0)
            guard let length = try f.read(upToCount: 8), length.count == 8 else { throw PrecisionError("Missing safetensors header") }
            let n = length.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
            guard n <= 64 << 20, n <= size - 8, let data = try f.read(upToCount: Int(n)), data.count == Int(n),
                let header = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { throw PrecisionError("Invalid safetensors header") }
            var out: [String: Entry] = [:]
            for (key, raw) in header where key != "__metadata__" {
                guard let item = raw as? [String: Any], let code = item["dtype"] as? String, let dtype = Self.types[code],
                    let shape = item["shape"] as? [Int], shape.allSatisfy({ $0 >= 0 }),
                    let offsets = item["data_offsets"] as? [UInt64], offsets.count == 2,
                    offsets[1] >= offsets[0], offsets[1] <= size - 8 - n
                else { throw PrecisionError("Invalid tensor descriptor: \(key)") }
                var count = dtype.size
                for dim in shape {
                    let (product, overflow) = count.multipliedReportingOverflow(by: dim)
                    guard !overflow else { throw PrecisionError("Tensor size overflow: \(key)") }; count = product
                }
                guard UInt64(count) == offsets[1] - offsets[0] else { throw PrecisionError("Tensor byte count mismatch: \(key)") }
                out[key] = Entry(dtype: dtype, shape: shape, offset: 8 + n + offsets[0], bytes: count)
            }
            var end: UInt64 = 8 + n
            for entry in out.values.sorted(by: { $0.offset == $1.offset ? $0.bytes < $1.bytes : $0.offset < $1.offset }) {
                guard entry.offset == end else { throw PrecisionError("Overlapping or non-contiguous tensors") }
                end += UInt64(entry.bytes)
            }
            guard end == size else { throw PrecisionError("Unindexed tensor payload") }
            self.file = f; self.entries = out
        } catch { try? f.close(); throw error }
    }
    deinit { try? file.close() }
    public func array(_ key: String) throws -> MLXArray {
        guard let e = entries[key] else { throw PrecisionError("Missing tensor: \(key)") }
        try file.seek(toOffset: e.offset)
        guard let data = try file.read(upToCount: e.bytes), data.count == e.bytes else { throw PrecisionError("Truncated tensor: \(key)") }
        return MLXArray(data, e.shape, dtype: e.dtype)
    }
    public func arrays() throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for key in entries.keys.sorted() { result[key] = try autoreleasepool { try array(key) } }
        return result
    }
}

/// Writes payloads as they arrive. At most one source tensor and its packed result are live during conversion.
public final class TensorWriter {
    private let destination: URL, bodyURL: URL
    private let body: FileHandle
    private var header: [String: Any] = [:]
    private var offset = 0
    public init(_ destination: URL) throws {
        self.destination = destination
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        bodyURL = destination.deletingLastPathComponent().appendingPathComponent(".intake-\(UUID().uuidString).body")
        FileManager.default.createFile(atPath: bodyURL.path, contents: nil)
        body = try FileHandle(forWritingTo: bodyURL)
    }
    deinit { try? body.close(); try? FileManager.default.removeItem(at: bodyURL) }
    public func append(_ key: String, _ array: MLXArray) throws {
        guard header[key] == nil, key != "__metadata__", let code = TensorArchive.types.first(where: { $0.value == array.dtype })?.key else {
            throw PrecisionError("Duplicate key or unsupported tensor: \(key)")
        }
        try Self.requireCapacity(at: bodyURL, bytes: Int64(array.nbytes))
        let bytes = array.asData(access: .noCopyIfContiguous).data
        header[key] = ["dtype": code, "shape": array.shape, "data_offsets": [offset, offset + bytes.count]]
        try body.write(contentsOf: bytes); offset += bytes.count
    }
    public func finish() throws {
        var data = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        data.append(Data(repeating: 32, count: (8 - data.count % 8) % 8))
        var length = UInt64(data.count).littleEndian
        let prefix = 8 + data.count
        try Self.requireCapacity(at: bodyURL, bytes: Int64(prefix))
        // Make room for the header in this SAME inode, moving backwards so overlapping chunks are safe.
        // Source + one output payload is the disk peak; finish never creates another payload-sized file.
        try body.truncate(atOffset: UInt64(offset + prefix))
        let input = try FileHandle(forReadingFrom: bodyURL); defer { try? input.close() }
        var end = offset
        while end > 0 {
            try autoreleasepool {
                let start = max(0, end - (4 << 20)), size = end - start
                try input.seek(toOffset: UInt64(start))
                guard let chunk = try input.read(upToCount: size), chunk.count == size else { throw PrecisionError("Truncated conversion payload") }
                try body.seek(toOffset: UInt64(start + prefix)); try body.write(contentsOf: chunk)
                end = start
            }
        }
        try body.seek(toOffset: 0)
        try withUnsafeBytes(of: &length) { try body.write(contentsOf: $0) }
        try body.write(contentsOf: data); try body.synchronize(); try body.close()
        guard rename(bodyURL.path, destination.path) == 0 else { throw PrecisionError("Cannot commit converted file") }
    }
    static func requireCapacity(at url: URL, bytes: Int64) throws {
        let attrs = try FileManager.default.attributesOfFileSystem(forPath: url.deletingLastPathComponent().path)
        guard let free = attrs[.systemFreeSize] as? NSNumber, free.int64Value >= bytes + (64 << 20) else {
            throw PrecisionError("Insufficient conversion staging capacity")
        }
    }

}

public enum StreamedPrecision {
    public struct Coverage: Codable, Sendable {
        public var tensors = 0, quantizedTensors = 0
        public var sourceBytes = 0, outputBytes = 0, largestTensorBytes = 0
        public init() {}
    }
    public static func convert(
        keys: [String], read: (String) throws -> MLXArray, to output: URL,
        half: DType, bits: Int? = nil, eligible: (String, [Int]) -> Bool
    ) throws -> Coverage {
        guard half == .float16 || half == .bfloat16, bits == nil || bits == 8 || bits == 4 else { throw PrecisionError("Use half precision and affine int8/int4") }
        guard !keys.contains(where: { $0.hasSuffix(".scales") || $0.hasSuffix(".biases") }) else {
            throw PrecisionError("Conversion requires original floating weights, never a packed source")
        }
        let writer = try TensorWriter(output)
        var coverage = Coverage()
        for key in keys.sorted() {
            try autoreleasepool {
                let source = try read(key)
                coverage.tensors += 1; coverage.sourceBytes += source.nbytes
                coverage.largestTensorBytes = max(coverage.largestTensorBytes, source.nbytes)
                let isFloat = [.float16, .bfloat16, .float32, .float64].contains(source.dtype)
                let value = isFloat ? source.asType(half, stream: .cpu) : source
                if isFloat, half == .float16 {
                    // Refuse range loss instead of silently writing infinities. A BF16 exception needs a quality gate.
                    let bad = MLX.any(.!MLX.isFinite(value), stream: .cpu)
                    guard !bad.item(Bool.self) else { throw PrecisionError("FP16 non-finite tensor: \(key)") }
                }
                if let bits, isFloat, value.ndim == 2, value.dim(1) % 64 == 0, eligible(key, value.shape) {
                    guard key.hasSuffix(".weight") else { throw PrecisionError("Quantized tensor must be a module weight: \(key)") }
                    let prefix = String(key.dropLast(7))
                    let (packed, scales, biases) = MLX.quantized(value, groupSize: 64, bits: bits, mode: .affine, stream: .cpu)
                    guard let biases else { throw PrecisionError("Missing affine biases") }
                    eval(packed, scales, biases)
                    try writer.append(key, packed); try writer.append(prefix + ".scales", scales); try writer.append(prefix + ".biases", biases)
                    coverage.quantizedTensors += 1; coverage.outputBytes += packed.nbytes + scales.nbytes + biases.nbytes
                } else {
                    eval(value); try writer.append(key, value); coverage.outputBytes += value.nbytes
                }
            }
            Memory.clearCache()
        }
        try writer.finish()
        return coverage
    }

    /// Install stored quantized modules directly, without quantizing random dense placeholders.
    public static func prepare(_ model: Module, weights: [String: MLXArray]) throws {
        let names = Set(weights.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(7)) })
        let packedNames = Set(weights.filter { $0.key.hasSuffix(".weight") && $0.value.dtype == .uint32 }.map { String($0.key.dropLast(7)) })
        guard packedNames == names else { throw PrecisionError("Packed weights and scale groups differ") }
        var updates: [(String, Module)] = []
        for (path, layer) in model.leafModules().flattened() where names.contains(path) {
            guard let w = weights[path + ".weight"], let s = weights[path + ".scales"], let b = weights[path + ".biases"],
                w.dtype == .uint32, w.ndim == 2, s.ndim == 2, b.shape == s.shape, b.dtype == s.dtype, [.float16, .bfloat16].contains(s.dtype), w.dim(0) == s.dim(0)
            else { throw PrecisionError("Invalid quantized group: \(path)") }
            let inputs: Int
            if let quantized = layer as? QuantizedLinear {
                inputs = quantized.weight.dim(1) * 32 / quantized.bits
            } else if let linear = layer as? Linear {
                inputs = linear.weight.dim(1)
            } else if let quantized = layer as? QuantizedEmbedding {
                inputs = quantized.weight.dim(1) * 32 / quantized.bits
            } else if let embedding = layer as? Embedding {
                inputs = embedding.weight.dim(1)
            } else {
                throw PrecisionError("Packed weights do not map to a Linear/Embedding: \(path)")
            }
            let rows = (layer as? Linear)?.weight.dim(0) ?? (layer as? Embedding)?.weight.dim(0) ?? 0
            guard w.dim(0) == rows, inputs > 0, s.dim(1) * 64 == inputs, w.dim(1) * 32 % inputs == 0 else { throw PrecisionError("Packed shape mismatch: \(path)") }
            let bits = w.dim(1) * 32 / inputs
            guard bits == 4 || bits == 8 else { throw PrecisionError("Unsupported packed bits: \(path)") }
            if let linear = layer as? Linear {
                updates.append((path, QuantizedLinear(weight: w, bias: linear.bias, scales: s, biases: b, groupSize: 64, bits: bits)))
            } else {
                updates.append((path, QuantizedEmbedding(weight: w, scales: s, biases: b, groupSize: 64, bits: bits)))
            }
        }
        guard updates.count == names.count else { throw PrecisionError("Quantized modules missing from model tree") }
        try model.update(modules: ModuleChildren.unflattened(updates), verify: .none)
    }
}
