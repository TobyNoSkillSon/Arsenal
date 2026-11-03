// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ModelPrecision
import ArsenalWire

/// Bounded reader for the pinned uncompressed PyTorch/NumPy ZIP containers (classic end record, stored entries).
/// It never deserializes pickle or invokes a Python/runtime subprocess; Get has verified each file's SHA-256.
final class C1Archive {
    struct Entry { let offset: Int; let size: Int }
    let file: FileHandle; let size: Int; var entries: [String: Entry] = [:]
    deinit { try? file.close() }
    init(_ url: URL) throws {
        file = try FileHandle(forReadingFrom: url); size = Int(try file.seekToEnd())
        guard size >= 22 else { throw audioFailure("Truncated Community-1 archive.") }
        let end = try read(size - 22, 22)
        guard end.u32(0) == 0x06054b50, end.u16(20) == 0 else { throw audioFailure("Unsupported ZIP footer.") }
        let centralSize = Int(end.u32(12)), centralOffset = Int(end.u32(16))
        guard centralSize <= 1024 * 1024 else { throw audioFailure("ZIP directory too large.") }
        let central = try read(centralOffset, centralSize); var i = 0
        while i < central.count {
            guard i + 46 <= central.count, central.u32(i) == 0x02014b50, central.u16(i + 10) == 0 else { throw audioFailure("Expected an uncompressed pinned archive.") }
            let n = Int(central.u16(i + 28)), x = Int(central.u16(i + 30)), c = Int(central.u16(i + 32))
            guard i + 46 + n + x + c <= central.count, let name = String(data: central.subdata(in: i + 46..<i + 46 + n), encoding: .utf8), entries[name] == nil else {
                throw audioFailure("Invalid ZIP entry.")
            }
            let offset = Int(central.u32(i + 42)), length = Int(central.u32(i + 24))
            let local = try read(offset, 30)
            guard local.u32(0) == 0x04034b50, local.u16(8) == 0 else { throw audioFailure("Invalid ZIP local header.") }
            let payload = offset + 30 + Int(local.u16(26)) + Int(local.u16(28))
            guard payload <= size, length <= size - payload else { throw audioFailure("ZIP entry exceeds archive.") }
            entries[name] = Entry(offset: payload, size: length); i += 46 + n + x + c
        }
    }
    func read(_ offset: Int, _ count: Int) throws -> Data {
        guard offset >= 0, count >= 0, count <= 64 * 1024 * 1024, offset <= size, count <= size - offset else { throw audioFailure("Invalid archive read.") }
        try file.seek(toOffset: UInt64(offset)); let data = try file.read(upToCount: count) ?? Data()
        guard data.count == count else { throw audioFailure("Truncated archive entry.") }; return data
    }
    func tensor(_ descriptor: C1TensorDescriptor) throws -> MLXArray {
        guard let entry = entries[descriptor.entry] else { throw audioFailure("Missing pinned tensor storage.") }
        let count = descriptor.shape.reduce(1, *), offset = descriptor.offset * 4
        guard offset >= 0, count * 4 <= entry.size - offset else { throw audioFailure("Tensor exceeds pinned storage.") }
        var array = MLXArray(try read(entry.offset + offset, count * 4), descriptor.shape, type: Float.self)
        if let permutation = descriptor.permutation { array = array.transposed(axes: permutation) }
        return array
    }
    func numpy(_ name: String) throws -> MLXArray {
        guard let entry = entries[name + ".npy"] else { throw audioFailure("Missing PLDA parameter: " + name) }
        let data = try read(entry.offset, entry.size)
        guard data.count >= 10, Array(data.prefix(6)) == [147, 78, 85, 77, 80, 89], data[6] == 1 else { throw audioFailure("Unsupported NumPy metadata.") }
        let h = Int(data.u16(8)), start = 10 + h
        guard start <= data.count, let header = String(data: data.subdata(in: 10..<start), encoding: .ascii), header.contains("'fortran_order': False") else {
            throw audioFailure("Unsupported NumPy layout.")
        }
        let pattern = "'shape': \\(([^)]*)\\)"
        let regex = try NSRegularExpression(pattern: pattern)
        guard let match = regex.firstMatch(in: header, range: NSRange(header.startIndex..., in: header)), let range = Range(match.range(at: 1), in: header) else {
            throw audioFailure("Missing NumPy shape.")
        }
        let shape = header[range].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let n = shape.reduce(1, *)
        if header.contains("'<f4'") {
            guard data.count - start == n * 4 else { throw audioFailure("PLDA size mismatch.") }
            return MLXArray(data.subdata(in: start..<data.count), shape, type: Float.self)
        }
        guard header.contains("'<f8'"), data.count - start == n * 8 else { throw audioFailure("Unsupported PLDA dtype.") }
        let floats: [Float] = (0..<n).map { Float(Double(bitPattern: data.u64(start + $0 * 8))) }
        return MLXArray(floats, shape)
    }
}
private extension Data {
    func u16(_ i: Int) -> UInt16 { UInt16(self[i]) | UInt16(self[i + 1]) << 8 }
    func u32(_ i: Int) -> UInt32 { UInt32(u16(i)) | UInt32(u16(i + 2)) << 16 }
    func u64(_ i: Int) -> UInt64 { UInt64(u32(i)) | UInt64(u32(i + 4)) << 32 }
}

/// Network matrices take the selected tier; the PLDA transform is host clustering data and stays half.
func c1Eligible(_ name: String, _ shape: [Int]) -> Bool {
    !name.hasPrefix("plda.") && audioEligible(name, shape)
}

/// Get prepares one half `model.safetensors` from the pinned PyTorch/NumPy containers (fixed descriptors, no
/// pickle execution); int8/int4 stream that prepared snapshot. Load never reads the original containers.
enum Community1Conversion {
    static let marker = "arsenal-community1-layout.json"
    static let legacyMarker = "mobius-community1-layout.json" // legacy-compat: read in snapshots written before the rename
    static let base = NativeConversion(recipe: "cast:audio-community1-v1", convert: prepare)
    static let tiers = NativeConversion(recipe: "audio-community1-v1", convert: derive)
    /// PLDA arrays: snapshot name → (npz container, array).
    static let pldaArrays: [(String, String, String)] = [
        ("plda.mean1", "xvec_transform.npz", "mean1"), ("plda.mean2", "xvec_transform.npz", "mean2"), ("plda.lda", "xvec_transform.npz", "lda"),
        ("plda.mu", "plda.npz", "mu"), ("plda.tr", "plda.npz", "tr"), ("plda.psi", "plda.npz", "psi")
    ]

    static func prepare(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> JSONObject {
        guard context.bits == nil else { throw PrecisionError("Community-1 base preparation requires a half tier") }
        let archives = try Dictionary(
            uniqueKeysWithValues: ["segmentation", "embedding"].map { ($0, try C1Archive(context.source.appendingPathComponent($0 + "/pytorch_model.bin"))) })
        let containers = try Dictionary(
            uniqueKeysWithValues: ["xvec_transform.npz", "plda.npz"].map { ($0, try C1Archive(context.source.appendingPathComponent("plda/" + $0))) })
        let descriptors = Dictionary(uniqueKeysWithValues: c1Descriptors.map { ($0.name, $0) })
        let plda = Dictionary(uniqueKeysWithValues: pldaArrays.map { ($0.0, ($0.1, $0.2)) })
        _ = try StreamedPrecision.convert(
            keys: (Array(descriptors.keys) + Array(plda.keys)).sorted(),
            read: { key in
                try cancellation.check()
                if let descriptor = descriptors[key] {
                    guard let archive = archives[descriptor.component] else { throw PrecisionError("Missing Community-1 component") }
                    return try archive.tensor(descriptor)
                }
                guard let (container, array) = plda[key], let archive = containers[container] else { throw PrecisionError("Unknown Community-1 tensor: " + key) }
                return try archive.numpy(array)
            }, to: context.destination.appendingPathComponent("model.safetensors"), half: context.half, eligible: { _, _ in false })
        try FileManager.default.copyItem(at: context.source.appendingPathComponent("config.yaml"), to: context.destination.appendingPathComponent("config.yaml"))
        try JSONSerialization.data(withJSONObject: ["architecture": "pyannote_community_1", "layout_version": 1], options: [.sortedKeys])
            .write(to: context.destination.appendingPathComponent(marker))
        return try context.receipt(cancellation: cancellation)
    }

    static func derive(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> JSONObject {
        guard context.bits != nil, FileManager.default.fileExists(atPath: StoredNames.current(marker, legacy: legacyMarker, in: context.source).path) else {
            throw PrecisionError("Prepare the Community-1 half snapshot before deriving int8/int4")
        }
        try context.convertTree(from: context.source, cancellation: cancellation, eligible: c1Eligible)
        return try context.receipt(cancellation: cancellation)
    }
}
