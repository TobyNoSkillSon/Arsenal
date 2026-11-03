import Foundation
import MLX

/// Reader for NumPy `.npz` archives as published by Stability AI in
/// `stabilityai/stable-audio-3-optimized/MLX/*.npz` (written by `np.savez`:
/// a ZIP of `.npy` members, stored uncompressed). mlx-swift at the family pin
/// loads only `.safetensors` / `.npy`, so this parses the ZIP directory and the
/// `.npy` headers itself and copies each member straight from a memory-mapped
/// file into an `MLXArray` (no intermediate `Data` copy of the whole file).
///
/// Supported: stored (method 0) members, ZIP64 sizes/offsets, `.npy` format
/// v1/v2/v3, little-endian `f2 f4 f8 i1 i2 i4 i8 u1 u2 u4 u8 b1` and bf16 stored
/// as `V2` is not produced by numpy and is rejected. Fortran-order arrays are
/// rejected (numpy never writes them for C-contiguous MLX exports).
public enum NPZError: Error, LocalizedError, Equatable {
    case notZip(String)
    case compressedMember(String)
    case badNpy(String)
    case unsupportedDType(String, String)

    public var errorDescription: String? {
        switch self {
        case .notZip(let m): return "npz: not a ZIP archive: \(m)"
        case .compressedMember(let n): return "npz: member \(n) is compressed (np.savez_compressed); only stored members are supported"
        case .badNpy(let m): return "npz: bad .npy member: \(m)"
        case .unsupportedDType(let n, let d): return "npz: member \(n) has unsupported dtype \(d)"
        }
    }
}

public struct NPZMember: Sendable {
    public let name: String          // without the trailing `.npy`
    public let dtype: DType
    public let shape: [Int]
    public let dataOffset: Int       // absolute byte offset of the array payload
    public let byteCount: Int
}

public final class NPZArchive {
    public let url: URL
    public let members: [String: NPZMember]
    private let data: Data           // memory-mapped

    public init(url: URL) throws {
        self.url = url
        self.data = try Data(contentsOf: url, options: [.alwaysMapped])
        self.members = try NPZArchive.parse(data, file: url.lastPathComponent)
    }

    public var keys: [String] { members.keys.sorted() }

    /// Load one member as an MLXArray (copied out of the mapping).
    public func array(_ name: String) throws -> MLXArray {
        guard let m = members[name] else { throw NPZError.badNpy("no member \(name)") }
        return data.withUnsafeBytes { raw -> MLXArray in
            let slice = UnsafeRawBufferPointer(rebasing: raw[m.dataOffset ..< m.dataOffset + m.byteCount])
            if m.byteCount == 0 {
                return MLXArray.zeros(m.shape, dtype: m.dtype)
            }
            return MLXArray(Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: slice.baseAddress!),
                                 count: m.byteCount, deallocator: .none),
                            m.shape, dtype: m.dtype)
        }
    }

    /// Load every member (optionally filtered) into a flat dictionary.
    public func arrays(where include: (String) -> Bool = { _ in true }) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for k in members.keys where include(k) {
            out[k] = try array(k)
        }
        return out
    }

    // MARK: - ZIP parsing

    private static func u16(_ d: UnsafeRawBufferPointer, _ o: Int) -> Int {
        Int(d.loadUnaligned(fromByteOffset: o, as: UInt16.self).littleEndian)
    }
    private static func u32(_ d: UnsafeRawBufferPointer, _ o: Int) -> Int {
        Int(d.loadUnaligned(fromByteOffset: o, as: UInt32.self).littleEndian)
    }
    private static func u64(_ d: UnsafeRawBufferPointer, _ o: Int) -> Int {
        Int(d.loadUnaligned(fromByteOffset: o, as: UInt64.self).littleEndian)
    }

    static func parse(_ data: Data, file: String) throws -> [String: NPZMember] {
        try data.withUnsafeBytes { d -> [String: NPZMember] in
            let n = d.count
            guard n >= 22 else { throw NPZError.notZip("\(file): too short") }
            // End of central directory: scan back over a possible comment (≤ 64 KiB).
            var eocd = -1
            var i = n - 22
            let floor = max(0, n - 22 - 65_535)
            while i >= floor {
                if u32(d, i) == 0x0605_4b50 { eocd = i; break }
                i -= 1
            }
            guard eocd >= 0 else { throw NPZError.notZip("\(file): no end-of-central-directory record") }
            var count = u16(d, eocd + 10)
            var cdSize = u32(d, eocd + 12)
            var cdOffset = u32(d, eocd + 16)
            // ZIP64 locator sits immediately before the EOCD.
            if eocd >= 20, u32(d, eocd - 20) == 0x0706_4b50 {
                let z64 = u64(d, eocd - 20 + 8)
                guard z64 + 56 <= n, u32(d, z64) == 0x0606_4b50 else {
                    throw NPZError.notZip("\(file): bad ZIP64 end record")
                }
                count = u64(d, z64 + 32)
                cdSize = u64(d, z64 + 40)
                cdOffset = u64(d, z64 + 48)
            }
            guard cdOffset + cdSize <= n else { throw NPZError.notZip("\(file): central directory out of range") }

            var members: [String: NPZMember] = [:]
            var p = cdOffset
            for _ in 0..<count {
                guard p + 46 <= n, u32(d, p) == 0x0201_4b50 else {
                    throw NPZError.notZip("\(file): bad central directory entry at \(p)")
                }
                let method = u16(d, p + 10)
                var compSize = u32(d, p + 20)
                var size = u32(d, p + 24)
                let nameLen = u16(d, p + 28)
                let extraLen = u16(d, p + 30)
                let commentLen = u16(d, p + 32)
                var localOffset = u32(d, p + 42)
                let nameBytes = UnsafeRawBufferPointer(rebasing: d[(p + 46) ..< (p + 46 + nameLen)])
                let name = String(decoding: nameBytes, as: UTF8.self)
                // ZIP64 extended information extra field (id 0x0001): values present
                // only for the fields that are 0xFFFFFFFF, in this order.
                var e = p + 46 + nameLen
                let eEnd = e + extraLen
                while e + 4 <= eEnd {
                    let id = u16(d, e), len = u16(d, e + 2)
                    if id == 0x0001 {
                        var q = e + 4
                        if size == 0xFFFF_FFFF { size = u64(d, q); q += 8 }
                        if compSize == 0xFFFF_FFFF { compSize = u64(d, q); q += 8 }
                        if localOffset == 0xFFFF_FFFF { localOffset = u64(d, q); q += 8 }
                    }
                    e += 4 + len
                }
                p = eEnd + commentLen

                guard name.hasSuffix(".npy") else { continue }
                let key = String(name.dropLast(4))
                guard method == 0 else { throw NPZError.compressedMember(key) }
                guard compSize == size else { throw NPZError.notZip("\(file): \(key) stored sizes differ") }
                guard localOffset + 30 <= n, u32(d, localOffset) == 0x0403_4b50 else {
                    throw NPZError.notZip("\(file): bad local header for \(key)")
                }
                let lName = u16(d, localOffset + 26)
                let lExtra = u16(d, localOffset + 28)
                let start = localOffset + 30 + lName + lExtra
                guard start + size <= n else { throw NPZError.notZip("\(file): \(key) payload out of range") }
                members[key] = try parseNpy(d, start: start, size: size, name: key)
            }
            return members
        }
    }

    // MARK: - .npy header

    static func parseNpy(_ d: UnsafeRawBufferPointer, start: Int, size: Int, name: String) throws -> NPZMember {
        guard size >= 10, d[start] == 0x93,
              String(decoding: UnsafeRawBufferPointer(rebasing: d[(start + 1) ..< (start + 6)]), as: UTF8.self) == "NUMPY"
        else { throw NPZError.badNpy("\(name): missing magic") }
        let major = Int(d[start + 6])
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = u16(d, start + 8); headerStart = start + 10
        } else {
            headerLen = u32(d, start + 8); headerStart = start + 12
        }
        guard headerStart + headerLen <= start + size else { throw NPZError.badNpy("\(name): header overruns member") }
        let header = String(decoding: UnsafeRawBufferPointer(rebasing: d[headerStart ..< (headerStart + headerLen)]), as: UTF8.self)
        let (descr, fortran, shape) = try parseHeaderDict(header, name: name)
        guard !fortran else { throw NPZError.badNpy("\(name): fortran_order arrays are not supported") }
        let dtype = try npyDType(descr, name: name)
        let count = shape.reduce(1, *)
        let payload = headerStart + headerLen
        let bytes = count * dtype.size
        guard payload + bytes <= start + size else {
            throw NPZError.badNpy("\(name): payload \(bytes) B exceeds member size")
        }
        return NPZMember(name: name, dtype: dtype, shape: shape, dataOffset: payload, byteCount: bytes)
    }

    /// Parse the Python-literal header dict: `{'descr': '<f2', 'fortran_order': False, 'shape': (3, 4), }`.
    static func parseHeaderDict(_ h: String, name: String) throws -> (String, Bool, [Int]) {
        func value(after key: String) -> Substring? {
            guard let r = h.range(of: "'\(key)':") else { return nil }
            return h[r.upperBound...].drop(while: { $0 == " " })
        }
        guard let dv = value(after: "descr"), let q = dv.first, q == "'" || q == "\"",
              let end = dv.dropFirst().firstIndex(of: q)
        else { throw NPZError.badNpy("\(name): no descr in header \(h)") }
        let descr = String(dv[dv.index(after: dv.startIndex) ..< end])
        guard let fv = value(after: "fortran_order") else { throw NPZError.badNpy("\(name): no fortran_order") }
        let fortran = fv.hasPrefix("True")
        guard let sv = value(after: "shape"), sv.first == "(", let close = sv.firstIndex(of: ")") else {
            throw NPZError.badNpy("\(name): no shape")
        }
        let inner = sv[sv.index(after: sv.startIndex) ..< close]
        let dims = try inner.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .map { s -> Int in
                guard let v = Int(s.hasSuffix("L") ? String(s.dropLast()) : s) else {
                    throw NPZError.badNpy("\(name): bad shape \(inner)")
                }
                return v
            }
        return (descr, fortran, dims)
    }

    static func npyDType(_ descr: String, name: String) throws -> DType {
        // Byte order: '<' little, '|' not applicable, '=' native (little on arm64).
        guard let order = descr.first, "<|=".contains(order) else {
            throw NPZError.unsupportedDType(name, descr)
        }
        switch descr.dropFirst() {
        case "f2": return .float16
        case "f4": return .float32
        case "f8": return .float64
        case "i1": return .int8
        case "i2": return .int16
        case "i4": return .int32
        case "i8": return .int64
        case "u1": return .uint8
        case "u2": return .uint16
        case "u4": return .uint32
        case "u8": return .uint64
        case "b1": return .bool
        default: throw NPZError.unsupportedDType(name, descr)
        }
    }
}
