import Foundation

/// A minimal safetensors writer/reader for converted weights (Von's fp32 checkpoint → fp16 at Get). Foundation only:
/// no MLX and no GPU, so conversion runs on the CPU and is unit-testable on tiny synthetic tensors.
/// Format: 8-byte little-endian header length, a JSON header {name: {dtype, shape, data_offsets}}, then the data.
public enum Safetensors {
    public struct Tensor: Equatable {
        public var name: String
        public var shape: [Int]
        /// "F16" or "F32".
        public var dtype: String
        public init(name: String, shape: [Int], dtype: String) { self.name = name; self.shape = shape; self.dtype = dtype }
        public var count: Int { shape.reduce(1, *) }
        public var bytes: Int { count * (["F16", "BF16"].contains(dtype) ? 2 : 4) }
    }

    /// Header bytes (length prefix included) for tensors laid out in order; padded with spaces to 8 bytes.
    public static func header(_ tensors: [Tensor], metadata: [String: String] = [:]) throws -> Data {
        var object: [String: Any] = [:]
        var offset = 0
        for t in tensors {
            object[t.name] = ["dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.bytes]]
            offset += t.bytes
        }
        if !metadata.isEmpty { object["__metadata__"] = metadata }
        var json = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        while (json.count + 8) % 8 != 0 { json.append(0x20) }
        var length = UInt64(json.count).littleEndian
        return Data(bytes: &length, count: 8) + json
    }

    /// IEEE float32 little-endian bytes → float16 bytes, round to nearest even (the cast MLX's astype performs).
    public static func float16(fromFloat32 data: Data) -> Data {
        let n = data.count / 4
        var out = Data(count: n * 2)
        data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
                let s = src.bindMemory(to: UInt32.self), d = dst.bindMemory(to: UInt16.self)
                for i in 0..<n { d[i] = Float16(Float(bitPattern: UInt32(littleEndian: s[i]))).bitPattern.littleEndian }
            }
        }
        return out
    }

    /// BF16 calibration casts never execute FP32 learned parameters.
    public static func bfloat16(fromFloat32 data: Data, truncate: Bool) -> Data {
        var out = Data(count: data.count / 2)
        data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
                let values = src.bindMemory(to: UInt32.self), target = dst.bindMemory(to: UInt16.self)
                for i in values.indices {
                    let bits = UInt32(littleEndian: values[i])
                    let rounded = truncate ? bits : bits &+ 0x7fff &+ ((bits >> 16) & 1)
                    target[i] = UInt16(truncatingIfNeeded: rounded >> 16).littleEndian
                }
            }
        }
        return out
    }

    /// Streams a file: `source(i)` returns tensor i's float32 bytes; tensors marked F16 are converted, F32 copied.
    /// Writes to `url` + ".partial" and renames on success. Returns the file size.
    @discardableResult
    public static func write(to url: URL, tensors: [Tensor], metadata: [String: String] = [:], source: (Int) throws -> Data) throws -> Int {
        let partial = url.appendingPathExtension("partial")
        try? FileManager.default.removeItem(at: partial)
        FileManager.default.createFile(atPath: partial.path, contents: nil)
        let handle = try FileHandle(forWritingTo: partial)
        var closed = false
        defer { if !closed { try? handle.close(); try? FileManager.default.removeItem(at: partial) } }
        let head = try header(tensors, metadata: metadata)
        try handle.write(contentsOf: head)
        var total = head.count
        for (i, t) in tensors.enumerated() {
            let raw = try source(i)
            guard raw.count == t.count * 4 else { throw VerdictError.message("\(t.name): expected \(t.count * 4) float32 bytes, got \(raw.count)") }
            let data = t.dtype == "F16" ? float16(fromFloat32: raw) : t.dtype == "BF16" ? bfloat16(fromFloat32: raw, truncate: metadata["rounding"] == "truncate") : raw
            try handle.write(contentsOf: data)
            total += data.count
        }
        try handle.close(); closed = true
        try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: partial, to: url)
        return total
    }

    /// Reads the header: tensors in data order. Throws on a malformed file.
    public static func tensors(at url: URL) throws -> [Tensor] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else { throw VerdictError.message("Truncated safetensors header") }
        let length = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard length < 100_000_000, let json = try handle.read(upToCount: Int(length)), json.count == Int(length),
              let object = try JSONSerialization.jsonObject(with: json) as? [String: Any] else { throw VerdictError.message("Invalid safetensors header") }
        let entries: [(Int, Tensor)] = try object.compactMap { name, value in
            guard name != "__metadata__" else { return nil }
            guard let v = value as? [String: Any], let dtype = v["dtype"] as? String, let shape = v["shape"] as? [Int],
                  let offsets = v["data_offsets"] as? [Int], offsets.count == 2 else { throw VerdictError.message("Invalid entry \(name)") }
            return (offsets[0], Tensor(name: name, shape: shape, dtype: dtype))
        }
        return entries.sorted { $0.0 < $1.0 }.map(\.1)
    }

    /// Tensor bytes as stored (tests and validation).
    public static func data(at url: URL, name: String) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        let length = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        let json = try handle.read(upToCount: Int(length)) ?? Data()
        guard let object = try JSONSerialization.jsonObject(with: json) as? [String: Any], let v = object[name] as? [String: Any],
              let offsets = v["data_offsets"] as? [Int] else { throw VerdictError.message("No tensor \(name)") }
        try handle.seek(toOffset: 8 + length + UInt64(offsets[0]))
        return try handle.read(upToCount: offsets[1] - offsets[0]) ?? Data()
    }
}

/// The manifest stored beside converted weights in Verdict's model store.
public struct StoredModelManifest: Codable, Equatable {
    public var model: String
    public var repository: String
    public var revision: String?
    /// The downloaded source file and its SHA-256 (the Hugging Face content hash when known).
    public var source_file: String
    public var source_bytes: Int64
    public var source_sha256: String?
    public var source_dtype: String
    /// What is stored: "fp16" (tensors listed in f32_tensors stay float32).
    public var dtype: String
    public var f32_tensors: [String]
    public var weights_file: String
    public var stored_bytes: Int64
    public var convert_seconds: Double
    public var converter: String
    public var created: String
    public init(model: String, repository: String, revision: String?, source_file: String, source_bytes: Int64, source_sha256: String?,
                source_dtype: String, dtype: String, f32_tensors: [String], weights_file: String, stored_bytes: Int64,
                convert_seconds: Double, converter: String, created: String) {
        self.model = model; self.repository = repository; self.revision = revision; self.source_file = source_file
        self.source_bytes = source_bytes; self.source_sha256 = source_sha256; self.source_dtype = source_dtype; self.dtype = dtype
        self.f32_tensors = f32_tensors; self.weights_file = weights_file; self.stored_bytes = stored_bytes
        self.convert_seconds = convert_seconds; self.converter = converter; self.created = created
    }
}
