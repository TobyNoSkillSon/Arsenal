// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
@testable import IntegrationKit

/// The shared PyTorch checkpoint reader (Audio, Sound and Restoration use it): a stored-ZIP `torch.save` file built in
/// the test, read plain and from inside an outer container, plus the pickle subset and its refusals.
struct TorchCheckpointTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    /// OrderedDict({"w": _rebuild_tensor_v2(storage("0", Float), 4, (2, 3), (3, 1), False, OrderedDict())}), protocol 2.
    static var statePickle: [UInt8] {
        var p: [UInt8] = [0x80, 0x02]
        p += [0x63] + Array("collections\nOrderedDict\n".utf8) + [0x29, 0x52]
        p += [0x58, 1, 0, 0, 0] + Array("w".utf8)
        p += [0x63] + Array("torch._utils\n_rebuild_tensor_v2\n".utf8)
        p += [0x28, 0x28, 0x58, 7, 0, 0, 0] + Array("storage".utf8) + [0x63] + Array("torch\nFloatStorage\n".utf8)
        p += [0x58, 1, 0, 0, 0] + Array("0".utf8) + [0x58, 3, 0, 0, 0] + Array("cpu".utf8) + [0x4B, 10, 0x74, 0x51]
        p += [0x4B, 4, 0x4B, 2, 0x4B, 3, 0x86, 0x4B, 3, 0x4B, 1, 0x86, 0x89]
        p += [0x63] + Array("collections\nOrderedDict\n".utf8) + [0x29, 0x52, 0x74, 0x52, 0x73, 0x2E]
        return p
    }

    /// A stored (uncompressed) ZIP with the given entries, in order; `method` 8 marks every entry compressed.
    static func zip(_ entries: [(String, [UInt8])], method: UInt16 = 0) -> [UInt8] {
        func le16(_ v: Int) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8 & 0xFF)] }
        func le32(_ v: Int) -> [UInt8] { le16(v & 0xFFFF) + le16(v >> 16 & 0xFFFF) }
        var out: [UInt8] = [], central: [UInt8] = []
        for (name, data) in entries {
            let offset = out.count, n = Array(name.utf8)
            out += le32(0x0403_4b50) + le16(20) + le16(0) + le16(Int(method)) + le16(0) + le16(0) + le32(0) + le32(data.count) + le32(data.count)
            out += le16(n.count) + le16(0) + n + data
            central += le32(0x0201_4b50) + le16(20) + le16(20) + le16(0) + le16(Int(method)) + le16(0) + le16(0) + le32(0) + le32(data.count)
            central += le32(data.count) + le16(n.count) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0) + le32(offset) + n
        }
        let start = out.count
        out += central
        out += le32(0x0605_4b50) + le16(0) + le16(0) + le16(entries.count) + le16(entries.count) + le32(central.count) + le32(start) + le16(0)
        return out
    }

    /// Storage "0": ten floats 0...9; the tensor starts at element 4.
    static var storage: [UInt8] { (0..<10).flatMap { withUnsafeBytes(of: Float($0).bitPattern.littleEndian, Array.init) } }

    static func write(_ bytes: [UInt8]) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("torch-ckpt-" + UUID().uuidString + ".pt")
        try Data(bytes).write(to: url)
        return url
    }

    @Test func readsAStateDictPlainAndInsideAContainer() throws {
        let archive = Self.zip([("archive/data.pkl", Self.statePickle), ("archive/data/0", Self.storage)])
        let plain = try Self.write(archive)
        defer { try? FileManager.default.removeItem(at: plain) }
        let checkpoint = try TorchCheckpoint(plain)
        let tensors = try checkpoint.tensors()
        #expect(tensors.count == 1)
        let w = try #require(tensors["w"])
        #expect(w == TorchCheckpoint.Tensor(storage: "0", dtype: .float32, offset: 4, shape: [2, 3], stride: [3, 1]))
        let values = try checkpoint.array(w).asArray(Float.self)
        #expect(values == [4, 5, 6, 7, 8, 9])
        // The same archive 100 bytes into an outer file (a .nemo tar member), read by byte range.
        let wrapped = try Self.write(Array(repeating: 0x55, count: 100) + archive + Array(repeating: 0xAA, count: 7))
        defer { try? FileManager.default.removeItem(at: wrapped) }
        let handle = try FileHandle(forReadingFrom: wrapped)
        let inner = try TorchCheckpoint(handle, base: 100, size: UInt64(archive.count))
        let innerW = try #require(try inner.tensors()["w"])
        let innerValues = try inner.array(innerW).asArray(Float.self)
        #expect(innerValues == [4, 5, 6, 7, 8, 9])
        #expect(throws: (any Error).self) { try TorchCheckpoint(handle, base: 100, size: UInt64(archive.count + 100)) }
    }

    @Test func refusesCompressedEntriesAndMissingPickle() throws {
        let compressed = try Self.write(Self.zip([("archive/data.pkl", Self.statePickle)], method: 8))
        let empty = try Self.write(Self.zip([("archive/data/0", Self.storage)]))
        defer { try? FileManager.default.removeItem(at: compressed); try? FileManager.default.removeItem(at: empty) }
        #expect(throws: (any Error).self) { try TorchCheckpoint(compressed) }
        #expect(throws: (any Error).self) { try TorchCheckpoint(empty) }
    }

    /// Bugs (2 Nov 2026 review): two crafted files trapped instead of failing. A last directory entry whose ZIP64 sizes
    /// are 0xFFFFFFFF but whose ZIP64 extra field is empty read past the directory; a zero-element tensor with huge
    /// dimensions overflowed the contiguity product.
    @Test func craftedArchivesAndShapesFailCleanly() throws {
        func le16(_ v: Int) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8 & 0xFF)] }
        func le32(_ v: Int) -> [UInt8] { le16(v & 0xFFFF) + le16(v >> 16 & 0xFFFF) }
        let name = Array("data.pkl".utf8)
        var archive = le32(0x0403_4b50) + le16(20) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0) + le32(0) + le32(0) + le16(name.count) + le16(0) + name
        let start = archive.count
        let central =
            le32(0x0201_4b50) + le16(20) + le16(20) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0) + le32(0xFFFF_FFFF)
            + le32(0xFFFF_FFFF) + le16(name.count) + le16(4) + le16(0) + le16(0) + le16(0) + le32(0) + le32(0) + name + le16(0x0001) + le16(0)
        archive += central + le32(0x0605_4b50) + le16(0) + le16(0) + le16(1) + le16(1) + le32(central.count) + le32(start) + le16(0)
        let crafted = try Self.write(archive)
        defer { try? FileManager.default.removeItem(at: crafted) }
        #expect(throws: (any Error).self) { try TorchCheckpoint(crafted) }

        let plain = try Self.write(Self.zip([("archive/data.pkl", Self.statePickle), ("archive/data/0", Self.storage)]))
        defer { try? FileManager.default.removeItem(at: plain) }
        let checkpoint = try TorchCheckpoint(plain)
        let huge = TorchCheckpoint.Tensor(storage: "0", dtype: .float32, offset: 0, shape: [0, 1 << 40, 1 << 40], stride: [1, 1 << 40, 1])
        #expect(throws: (any Error).self) { try checkpoint.array(huge) }
        let wide = TorchCheckpoint.Tensor(storage: "0", dtype: .float32, offset: 0, shape: [0, 1 << 33], stride: [1 << 33, 1])
        #expect(throws: (any Error).self) { try checkpoint.array(wide) }
    }

    @Test func pickleSubsetKeepsCallablesOpaque() throws {
        // An unknown callable stays a named placeholder; NEWOBJ (a Lightning checkpoint's dataclasses) too.
        #expect(try TorchPickle.parse(Data([0x80, 0x02, 0x63] + Array("os\nsystem\n".utf8) + [0x29, 0x52, 0x2E])) == .opaque("os.system"))
        let newobj: [UInt8] = [0x80, 0x02, 0x7D, 0x58, 4, 0, 0, 0] + Array("spec".utf8) + [0x63] + Array("m\nC\n".utf8) + [0x29, 0x81, 0x73, 0x2E]
        guard case .dict(let box) = try TorchPickle.parse(Data(newobj)) else { Issue.record("not a dictionary"); return }
        #expect(box.items.count == 1 && box.items[0].1 == .opaque("m.C"))
        // Hostile input fails cleanly: deep nesting, STOP on an empty stack, an unknown memo, an unknown opcode.
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data([0x80, 0x02, 0x4E] + Array(repeating: 0x85, count: 40) + [0x2E])) }
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data([0x80, 0x02, 0x2E])) }
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data([0x80, 0x02, 0x68, 0x05, 0x2E])) }
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data([0x80, 0x02, 0x69, 0x2E])) }
    }

    /// A hostile pickle cannot hang or crash the reader: nesting is measured once per value (a tuple of 30 levels built
    /// from one shared value holds 2^30 leaves), lists and dicts are depth-limited and cannot grow once placed inside
    /// another value, the memo is bounded, and container keys are not compared deeply.
    @Test func picklesWithSharedOrDeepValuesFailFastOrParse() throws {
        func shared(_ levels: Int) -> [UInt8] { [0x4E] + Array(repeating: [0x71, 0, 0x68, 0, 0x86], count: levels).flatMap { $0 } }
        guard case .tuple = try TorchPickle.parse(Data([0x80, 0x02] + shared(30) + [0x2E])) else { Issue.record("not a tuple"); return }
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data([0x80, 0x02] + shared(33) + [0x2E])) }
        // Two equal 30-level keys in one dict: no deep comparison.
        let keys: [UInt8] = [0x80, 0x02, 0x7D] + shared(30) + [0x4E, 0x73] + shared(30) + [0x4E, 0x73, 0x2E]
        guard case .dict(let box) = try TorchPickle.parse(Data(keys)) else { Issue.record("not a dictionary"); return }
        #expect(box.items.count == 2)
        // Lists nested by APPEND: 20 levels parse, 40 are refused.
        func nested(_ levels: Int) -> [UInt8] { Array(repeating: 0x5D, count: levels) + Array(repeating: 0x61, count: levels - 1) + [0x2E] }
        _ = try TorchPickle.parse(Data([0x80, 0x02] + nested(20)))
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data([0x80, 0x02] + nested(40))) }
        // A list already inside another cannot be reopened through the memo to deepen it unseen.
        let reopened: [UInt8] = [0x80, 0x02, 0x5D, 0x5D, 0x71, 1, 0x61, 0x68, 1, 0x5D, 0x61, 0x2E]
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data(reopened)) }
        // A recursive list (pickle's own form for a list that holds itself) still parses.
        let recursive: [UInt8] = [0x80, 0x02, 0x5D, 0x71, 0, 0x28, 0x68, 0, 0x65, 0x2E]
        guard case .list(let selfList) = try TorchPickle.parse(Data(recursive)) else { Issue.record("not a list"); return }
        #expect(selfList.items.count == 1)
        selfList.items.removeAll()
        // The memo is bounded even when nothing new is pushed.
        var memo: [UInt8] = [0x80, 0x02, 0x4E]
        for key in 0...1_000_000 { memo += [0x72, UInt8(key & 0xFF), UInt8(key >> 8 & 0xFF), UInt8(key >> 16 & 0xFF), 0] }
        #expect(throws: (any Error).self) { try TorchPickle.parse(Data(memo + [0x2E])) }
        // A dict of 200,000 int keys builds in linear time, and a repeated int key still replaces its value.
        var ints: [UInt8] = [0x80, 0x02, 0x7D, 0x28]
        for key in 0..<200_000 { ints += [0x4A, UInt8(key & 0xFF), UInt8(key >> 8 & 0xFF), UInt8(key >> 16 & 0xFF), 0, 0x4E] }
        ints += [0x75, 0x4B, 5, 0x88, 0x73, 0x2E]
        guard case .dict(let intDict) = try TorchPickle.parse(Data(ints)) else { Issue.record("not a dictionary"); return }
        #expect(intDict.items.count == 200_000 && intDict.items[5].1 == .bool(true))
        // BINFLOAT is big-endian, as pickle writes it.
        #expect(try TorchPickle.parse(Data([0x80, 0x02, 0x47, 0x3F, 0xF8, 0, 0, 0, 0, 0, 0, 0x2E])) == .float(1.5))
    }

    /// Release review E (M-E3): a memoized NaN fetched back as a key 200,000 times is one entry, as Python loads it, found
    /// in one lookup each time (keyed on its bits, NaN never equalling itself put every insert in one bucket: quadratic);
    /// -0.0 and 0.0 stay one key. A dict key over 64 KB is refused (L-E13: each lookup would hash it again).
    @Test func nanAndHugeDictKeysStayLinear() throws {
        let nan: [UInt8] = [0x47, 0x7F, 0xF8, 0, 0, 0, 0, 0, 0]
        var bytes: [UInt8] = [0x80, 0x02, 0x7D, 0x28] + nan + [0x71, 0, 0x4E]
        for _ in 0..<200_000 { bytes += [0x68, 0, 0x4E] }
        bytes += [0x75, 0x2E]
        let start = Date()
        guard case .dict(let box) = try TorchPickle.parse(Data(bytes)) else { Issue.record("not a dictionary"); return }
        #expect(box.items.count == 1)
        #expect(Date().timeIntervalSince(start) < 10, "200,000 NaN keys parse in linear time")
        let zeros: [UInt8] = [0x80, 0x02, 0x7D, 0x28, 0x47, 0, 0, 0, 0, 0, 0, 0, 0, 0x4E, 0x47, 0x80, 0, 0, 0, 0, 0, 0, 0, 0x88, 0x75, 0x2E]
        guard case .dict(let signed) = try TorchPickle.parse(Data(zeros)) else { Issue.record("not a dictionary"); return }
        #expect(signed.items.count == 1 && signed.items[0].1 == .bool(true))
        func keyed(_ length: Int) -> Data {
            Data([0x80, 0x02, 0x7D, 0x58] + withUnsafeBytes(of: UInt32(length).littleEndian, Array.init) + Array(repeating: 0x61, count: length) + [0x4E, 0x73, 0x2E])
        }
        guard case .dict = try TorchPickle.parse(keyed(65_536)) else { Issue.record("a 64 KB key is refused"); return }
        #expect(throws: (any Error).self) { try TorchPickle.parse(keyed(65_537)) }
    }

}
