// SPDX-License-Identifier: MIT
// Arsenal new-runner kit: non-executing readers for the checkpoint formats Get converts to safetensors. Do not edit.
//
// `convert` (Wire.swift) turns a pinned PyTorch `torch.save` archive (.pth, .pt) into `<name>.safetensors` with every
// floating tensor cast like a safetensors source, and a NumPy array file (.npy, .npz) into `<name>.safetensors` with each
// array stored losslessly in its own dtype. Nothing here imports, calls or evaluates anything a file names: the pickle is
// walked opcode by opcode, and only the reductions a state dict needs are interpreted. The reader follows the Audio
// family's reviewed TorchCheckpoint (Workers/Audio/Sources/AudioWorker/TorchCheckpoint.swift).
import Foundation
import MLX

/// A PyTorch ZIP checkpoint (stored entries, classic or ZIP64 directory).
final class TorchCheckpoint {
    struct Tensor: Equatable {
        let storage: String
        let dtype: DType
        let offset: Int
        let shape: [Int]
        let stride: [Int]
    }
    private let archive: StoredZip
    private let prefix: String
    /// The unpickled root object.
    let root: TorchPickle.Value

    init(_ url: URL) throws {
        archive = try StoredZip(url)
        guard let pickle = archive.names.filter({ $0.hasSuffix("/data.pkl") || $0 == "data.pkl" }).sorted().first else {
            throw RunnerError.failed("\(url.lastPathComponent) has no data.pkl: not a torch.save archive.")
        }
        prefix = String(pickle.dropLast("data.pkl".count))
        root = try TorchPickle.parse(try archive.read(pickle, limit: 64 << 20))
    }

    /// Every tensor in the checkpoint by name. Nested dictionaries (a state dict under "model" or "state_dict", say)
    /// are flattened with dotted names; values that are not tensors or dictionaries are skipped.
    func tensors() throws -> [String: Tensor] {
        var out: [String: Tensor] = [:]
        func walk(_ value: TorchPickle.Value, _ path: String, depth: Int) throws {
            guard depth <= 16 else { throw RunnerError.failed("Checkpoint dictionaries are nested too deeply.") }
            guard case .dict(let box) = value else { return }
            for (key, item) in box.items {
                let name: String
                switch key {
                case .string(let text): name = text
                case .int(let number): name = String(number)
                default: continue
                }
                let full = path.isEmpty ? name : path + "." + name
                switch item {
                case .tensor(let tensor):
                    guard out[full] == nil else { throw RunnerError.failed("Checkpoint names \(full) twice.") }
                    out[full] = tensor
                case .dict: try walk(item, full, depth: depth + 1)
                default: continue
                }
            }
        }
        try walk(root, "", depth: 0)
        return out
    }

    /// One tensor as an MLX array of its stored dtype (contiguous tensors only).
    func array(_ tensor: Tensor) throws -> MLXArray {
        // MLX has no float64 on the GPU; such a tensor is refused here, before any MLX call, rather than half-converted.
        guard tensor.dtype != .float64 else {
            throw RunnerError.failed("Storage \(tensor.storage) holds float64 values, which the converter does not cast; convert that checkpoint to float32 or 16-bit first.")
        }
        let name = prefix + "data/" + tensor.storage
        guard let size = archive.size(of: name) else { throw RunnerError.failed("Missing tensor storage \(tensor.storage).") }
        var count = 1
        for dim in tensor.shape {
            let (product, overflow) = count.multipliedReportingOverflow(by: dim)
            guard dim >= 0, !overflow else { throw RunnerError.failed("Invalid tensor shape \(tensor.shape).") }
            count = product
        }
        var contiguous = 1
        for (dim, stride) in zip(tensor.shape, tensor.stride).reversed() where dim > 1 {
            guard stride == contiguous else { throw RunnerError.failed("Non-contiguous checkpoint tensor (storage \(tensor.storage)).") }
            let (product, overflow) = contiguous.multipliedReportingOverflow(by: dim)
            guard !overflow else { throw RunnerError.failed("Invalid tensor shape \(tensor.shape).") }
            contiguous = product
        }
        let width = tensor.dtype.size
        let (start, startOverflow) = tensor.offset.multipliedReportingOverflow(by: width)
        let (bytes, bytesOverflow) = count.multipliedReportingOverflow(by: width)
        guard tensor.offset >= 0, !startOverflow, !bytesOverflow, start <= size, bytes <= size - start else {
            throw RunnerError.failed("Tensor exceeds its storage \(tensor.storage).")
        }
        let data = try archive.read(name, from: start, count: bytes)
        return MLXArray(data, tensor.shape, dtype: tensor.dtype)
    }
}

/// The pickle subset `torch.save` writes for state dicts (protocol 2). Values that need code to construct are kept as
/// named placeholders and never executed. Every structure a hostile file could use to exhaust time or memory is bounded
/// as it is built: nesting depth is carried in each value (never recomputed), a container is frozen once it is nested
/// (so depths stay exact and no container can hold itself), dictionary keys are strings or integers looked up in O(1),
/// the memo and the number of values are capped on every write, and marks live on their own stack.
enum TorchPickle {
    /// Deepest nesting of tuples, lists and dictionaries accepted (a state dict needs about 6).
    static let maxDepth = 64
    static let maxValues = 8_000_000
    static let maxMemo = 1_000_000

    final class DictBox {
        var items: [(Value, Value)] = []
        /// Position of each key, so a large state dict builds in linear time.
        var strings: [String: Int] = [:]
        var ints: [Int: Int] = [:]
        var depth = 1
        var frozen = false
    }
    final class ListBox {
        var items: [Value] = []
        var depth = 1
        var frozen = false
    }
    indirect enum Value {
        case none, bool(Bool), int(Int), float(Double), string(String), bytes(Data)
        case tuple([Value], depth: Int), list(ListBox), dict(DictBox)
        case global(String), storage(key: String, dtype: DType), tensor(TorchCheckpoint.Tensor), opaque(String)

        /// Nesting depth of containers (0 for a scalar).
        var depth: Int {
            switch self {
            case .tuple(_, let depth): return depth
            case .list(let box): return box.depth
            case .dict(let box): return box.depth
            default: return 0
            }
        }
    }
    static let storageTypes: [String: DType] = [
        "torch.FloatStorage": .float32, "torch.HalfStorage": .float16, "torch.BFloat16Storage": .bfloat16,
        "torch.DoubleStorage": .float64, "torch.LongStorage": .int64, "torch.IntStorage": .int32,
        "torch.ShortStorage": .int16, "torch.CharStorage": .int8, "torch.ByteStorage": .uint8, "torch.BoolStorage": .bool,
    ]

    static func parse(_ data: Data) throws -> Value {
        let bytes = [UInt8](data)
        var i = 0, stack: [Value] = [], marks: [Int] = [], memo: [Int: Value] = [:], pushed = 0
        func fail(_ message: String) -> RunnerError { RunnerError.failed("Checkpoint pickle: " + message) }
        func push(_ value: Value) throws {
            pushed += 1
            guard pushed <= maxValues else { throw fail("too many values") }
            stack.append(value)
        }
        func remember(_ key: Int) throws {
            guard let value = stack.last, stack.count > (marks.last ?? 0) else { throw fail("memo of nothing") }
            guard memo[key] != nil || memo.count < maxMemo else { throw fail("too many memo entries") }
            memo[key] = value
        }
        /// A value placed inside a tuple, list or dictionary: a container freezes, so its depth can no longer change.
        func nest(_ value: Value) {
            switch value {
            case .list(let box): box.frozen = true
            case .dict(let box): box.frozen = true
            default: break
            }
        }
        func tuple(_ items: [Value]) throws -> Value {
            let depth = 1 + (items.map(\.depth).max() ?? 0)
            guard depth <= maxDepth else { throw fail("nesting too deep") }
            items.forEach(nest)
            return .tuple(items, depth: depth)
        }
        func take(_ n: Int) throws -> ArraySlice<UInt8> {
            guard n >= 0, n <= bytes.count - i else { throw fail("truncated") }
            defer { i += n }
            return bytes[i ..< (i + n)]
        }
        func uint(_ n: Int) throws -> Int { try take(n).reversed().reduce(0) { $0 << 8 | Int($1) } }
        func pop() throws -> Value {
            guard stack.count > (marks.last ?? 0), let value = stack.popLast() else { throw fail("stack underflow") }
            return value
        }
        func popMark() throws -> [Value] {
            guard let index = marks.popLast(), index <= stack.count else { throw fail("missing mark") }
            let items = Array(stack[index...]); stack.removeSubrange(index...)
            return items
        }
        func line() throws -> String {
            guard let end = bytes[i...].firstIndex(of: 0x0A) else { throw fail("unterminated line") }
            defer { i = end + 1 }
            return String(decoding: bytes[i ..< end], as: UTF8.self)
        }
        func target() throws -> Value {
            guard let value = stack.last, stack.count > (marks.last ?? 0) else { throw fail("no container to fill") }
            return value
        }
        func setItems(_ items: [Value], into target: Value) throws {
            guard items.count % 2 == 0, case .dict(let box) = target else { throw fail("SETITEMS needs a dict") }
            guard !box.frozen else { throw fail("a dictionary changes after it was nested") }
            for index in stride(from: 0, to: items.count, by: 2) {
                let key = items[index], value = items[index + 1]
                let existing: Int?
                switch key {
                case .string(let name): existing = box.strings[name]
                case .int(let number): existing = box.ints[number]
                default: throw fail("a dictionary key that is not a string or an integer")
                }
                let depth = 1 + value.depth
                guard depth <= maxDepth else { throw fail("nesting too deep") }
                nest(value)
                guard !box.frozen else { throw fail("a dictionary changes after it was nested") }  // it held itself
                box.depth = Swift.max(box.depth, depth)
                if let existing {
                    box.items[existing].1 = value
                } else {
                    switch key {
                    case .string(let name): box.strings[name] = box.items.count
                    case .int(let number): box.ints[number] = box.items.count
                    default: break
                    }
                    box.items.append((key, value))
                }
            }
        }
        func append(_ items: [Value], into target: Value) throws {
            guard case .list(let box) = target else { throw fail("APPEND needs a list") }
            guard !box.frozen else { throw fail("a list changes after it was nested") }
            for value in items {
                let depth = 1 + value.depth
                guard depth <= maxDepth else { throw fail("nesting too deep") }
                nest(value)
                guard !box.frozen else { throw fail("a list changes after it was nested") }  // it held itself
                box.depth = Swift.max(box.depth, depth)
                box.items.append(value)
            }
        }
        func integers(_ value: Value, _ what: String) throws -> [Int] {
            guard case .tuple(let items, _) = value else { throw fail("malformed tensor \(what)") }
            return try items.map { (item: Value) -> Int in
                guard case .int(let v) = item else { throw fail("malformed tensor \(what)") }
                return v
            }
        }
        func reduce(_ callable: Value, _ arguments: Value) throws -> Value {
            guard case .global(let name) = callable, case .tuple(let args, _) = arguments else { return .opaque("call") }
            switch name {
            case "collections.OrderedDict": return .dict(DictBox())
            case "torch._utils._rebuild_parameter":
                return args.first ?? .none
            case "torch._utils._rebuild_tensor_v2":
                guard args.count >= 4, case .storage(let key, let dtype) = args[0], case .int(let offset) = args[1] else { throw fail("malformed tensor") }
                let shape = try integers(args[2], "shape"), strides = try integers(args[3], "stride")
                guard shape.count == strides.count else { throw fail("rank") }
                return .tensor(TorchCheckpoint.Tensor(storage: key, dtype: dtype, offset: offset, shape: shape, stride: strides))
            default: return .opaque(name)
            }
        }
        while true {
            guard i < bytes.count else { throw fail("missing STOP") }
            let op = bytes[i]; i += 1
            switch op {
            case 0x80: _ = try take(1)  // PROTO
            case 0x95: _ = try take(8)  // FRAME
            case 0x2E:  // STOP
                guard marks.isEmpty, stack.count == 1 else { throw fail("STOP with values left over") }
                return try pop()
            case 0x28:  // MARK
                guard marks.count < maxDepth * 4 else { throw fail("too many open marks") }
                marks.append(stack.count)
            case 0x7D: try push(.dict(DictBox()))  // EMPTY_DICT
            case 0x5D: try push(.list(ListBox()))  // EMPTY_LIST
            case 0x29: try push(.tuple([], depth: 1))  // EMPTY_TUPLE
            case 0x4E: try push(.none)  // NONE
            case 0x88: try push(.bool(true))  // NEWTRUE
            case 0x89: try push(.bool(false))  // NEWFALSE
            case 0x4B: try push(.int(try uint(1)))  // BININT1
            case 0x4D: try push(.int(try uint(2)))  // BININT2
            case 0x4A:  // BININT (signed 32-bit)
                try push(.int(Int(Int32(truncatingIfNeeded: UInt32(try uint(4))))))
            case 0x8A:  // LONG1
                let n = try uint(1)
                guard n <= 8 else { throw fail("long too wide") }
                let raw = try take(n)
                var value = 0
                for (k, byte) in raw.enumerated() { value |= Int(byte) << (8 * k) }
                if n > 0, n < 8, raw.last! & 0x80 != 0 { value -= 1 << (8 * n) }
                try push(.int(value))
            case 0x47:  // BINFLOAT
                let raw = try uint(8)
                try push(.float(Double(bitPattern: UInt64(bitPattern: Int64(raw)))))
            case 0x58: try push(.string(String(decoding: try take(try uint(4)), as: UTF8.self)))  // BINUNICODE
            case 0x8C: try push(.string(String(decoding: try take(try uint(1)), as: UTF8.self)))  // SHORT_BINUNICODE
            case 0x43: try push(.bytes(Data(try take(try uint(1)))))  // SHORT_BINBYTES
            case 0x42: try push(.bytes(Data(try take(try uint(4)))))  // BINBYTES
            case 0x55: try push(.string(String(decoding: try take(try uint(1)), as: UTF8.self)))  // SHORT_BINSTRING
            case 0x54: try push(.string(String(decoding: try take(try uint(4)), as: UTF8.self)))  // BINSTRING
            case 0x63:  // GLOBAL module\nname\n
                let module = try line(), name = try line()
                try push(.global(module + "." + name))
            case 0x93:  // STACK_GLOBAL
                let name = try pop(), module = try pop()
                guard case .string(let n) = name, case .string(let m) = module else { throw fail("STACK_GLOBAL") }
                try push(.global(m + "." + n))
            case 0x71: try remember(try uint(1))  // BINPUT
            case 0x72: try remember(try uint(4))  // LONG_BINPUT
            case 0x94: try remember(memo.count)  // MEMOIZE
            case 0x68:  // BINGET
                guard let value = memo[try uint(1)] else { throw fail("memo") }
                try push(value)
            case 0x6A:  // LONG_BINGET
                guard let value = memo[try uint(4)] else { throw fail("memo") }
                try push(value)
            case 0x74: try push(try tuple(try popMark()))  // TUPLE
            case 0x85: try push(try tuple([try pop()]))  // TUPLE1
            case 0x86:  // TUPLE2
                let b = try pop(), a = try pop(); try push(try tuple([a, b]))
            case 0x87:  // TUPLE3
                let c = try pop(), b = try pop(), a = try pop(); try push(try tuple([a, b, c]))
            case 0x52:  // REDUCE
                let arguments = try pop(), callable = try pop()
                try push(try reduce(callable, arguments))
            case 0x51:  // BINPERSID
                guard case .tuple(let id, _) = try pop(), id.count >= 3, case .string("storage") = id[0], case .global(let type) = id[1],
                    case .string(let key) = id[2]
                else { throw fail("unsupported persistent id") }
                guard let dtype = storageTypes[type] else { throw fail("unsupported storage \(type)") }
                try push(.storage(key: key, dtype: dtype))
            case 0x62: _ = try pop()  // BUILD: the state (OrderedDict `_metadata`) is not needed
            case 0x73:  // SETITEM
                let value = try pop(), key = try pop()
                try setItems([key, value], into: try target())
            case 0x75:  // SETITEMS
                let items = try popMark()
                try setItems(items, into: try target())
            case 0x61:  // APPEND
                let value = try pop()
                try append([value], into: try target())
            case 0x65:  // APPENDS
                let items = try popMark()
                try append(items, into: try target())
            default: throw fail(String(format: "unsupported opcode 0x%02x", op))
            }
        }
    }
}

/// NumPy `.npy` arrays and `.npz` archives of them (stored entries, as `np.savez` writes), little-endian, C order,
/// numeric dtypes only. An object array would need pickle, so it is refused.
enum NumpyArrays {
    struct Array { let dtype: String; let shape: [Int]; let data: Data }
    /// numpy descr -> safetensors dtype and element width.
    static let types: [String: (String, Int)] = [
        "<f8": ("F64", 8), "<f4": ("F32", 4), "<f2": ("F16", 2), "<i8": ("I64", 8), "<i4": ("I32", 4), "<i2": ("I16", 2),
        "|i1": ("I8", 1), "<u8": ("U64", 8), "<u4": ("U32", 4), "<u2": ("U16", 2), "|u1": ("U8", 1), "|b1": ("BOOL", 1),
    ]

    static func npy(_ data: Data, name: String) throws -> Array {
        let bytes = [UInt8](data.prefix(12))
        guard bytes.count >= 10, bytes[0] == 0x93, Data(bytes[1 ..< 6]) == Data("NUMPY".utf8) else {
            throw RunnerError.failed("\(name) is not a .npy array.")
        }
        let major = bytes[6]
        let headerLength: Int, start: Int
        switch major {
        case 1: headerLength = Int(bytes[8]) | Int(bytes[9]) << 8; start = 10
        case 2, 3:
            guard bytes.count >= 12 else { throw RunnerError.failed("\(name): truncated header.") }
            headerLength = Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16 | Int(bytes[11]) << 24; start = 12
        default: throw RunnerError.failed("\(name): .npy version \(major) is not supported.")
        }
        guard headerLength <= 1 << 16, start + headerLength <= data.count else { throw RunnerError.failed("\(name): invalid header.") }
        let header = String(decoding: data.subdata(in: (data.startIndex + start) ..< (data.startIndex + start + headerLength)), as: UTF8.self)
        func field(_ key: String) -> String? {
            guard let range = header.range(of: "'\(key)':") else { return nil }
            return String(header[range.upperBound...]).trimmingCharacters(in: .whitespaces)
        }
        guard let descrField = field("descr"), descrField.hasPrefix("'"),
            let descr = descrField.dropFirst().split(separator: "'", maxSplits: 1).first.map(String.init)
        else { throw RunnerError.failed("\(name): no dtype in the header.") }
        guard let (dtype, width) = types[descr] else {
            throw RunnerError.failed("\(name) holds dtype \(descr); only little-endian numeric arrays convert (object arrays need pickle).")
        }
        guard let order = field("fortran_order"), order.hasPrefix("False") else {
            throw RunnerError.failed("\(name) is stored in Fortran order; only C order converts.")
        }
        guard let shapeField = field("shape"), shapeField.hasPrefix("("), let close = shapeField.firstIndex(of: ")") else {
            throw RunnerError.failed("\(name): no shape in the header.")
        }
        let shape = try shapeField[shapeField.index(after: shapeField.startIndex) ..< close].split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            .map { (text: String) -> Int in
                guard let value = Int(text), value >= 0 else { throw RunnerError.failed("\(name): bad shape \(shapeField).") }
                return value
            }
        var count = 1
        for dim in shape {
            let (product, overflow) = count.multipliedReportingOverflow(by: dim)
            guard !overflow else { throw RunnerError.failed("\(name): shape overflows.") }
            count = product
        }
        let offset = start + headerLength
        let (expected, overflow) = count.multipliedReportingOverflow(by: width)
        guard !overflow, data.count - offset == expected else {
            throw RunnerError.failed("\(name) holds \(data.count - offset) data bytes; its header's shape \(shape) does not match.")
        }
        return Array(dtype: dtype, shape: shape, data: data.subdata(in: (data.startIndex + offset) ..< data.endIndex))
    }

    /// Every array of a .npy (named "array") or .npz file (named by member, without ".npy").
    static func read(_ url: URL) throws -> [String: Array] {
        if url.pathExtension == "npy" {
            return ["array": try npy(try Data(contentsOf: url), name: url.lastPathComponent)]
        }
        let archive = try StoredZip(url)
        var out: [String: Array] = [:]
        for member in archive.names.sorted() {
            guard member.hasSuffix(".npy") else { throw RunnerError.failed("\(url.lastPathComponent) holds \(member), not a .npy array.") }
            out[String(member.dropLast(4))] = try npy(try archive.read(member, limit: 1 << 31), name: member)
        }
        guard !out.isEmpty else { throw RunnerError.failed("\(url.lastPathComponent) holds no arrays.") }
        return out
    }

    /// A safetensors file written byte for byte from the arrays (no cast).
    static func writeSafetensors(_ arrays: [String: Array], metadata: [String: String], to url: URL) throws {
        var header: [String: Any] = ["__metadata__": metadata]
        var offset = 0
        let names = arrays.keys.sorted()
        for name in names {
            let array = arrays[name]!
            header[name] = ["dtype": array.dtype, "shape": array.shape, "data_offsets": [offset, offset + array.data.count]]
            offset += array.data.count
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw RunnerError.failed("Could not create \(url.path).") }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.write(contentsOf: withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) })
        try handle.write(contentsOf: json)
        for name in names { try handle.write(contentsOf: arrays[name]!.data) }
    }
}

/// A ZIP archive whose entries are stored uncompressed (torch.save and np.savez write these), read by offset.
final class StoredZip {
    private struct Entry { let offset: UInt64; let size: Int }
    private let file: FileHandle
    private let size: UInt64
    private var entries: [String: Entry] = [:]
    var names: [String] { Array(entries.keys) }
    deinit { try? file.close() }

    init(_ url: URL) throws {
        file = try FileHandle(forReadingFrom: url)
        size = try file.seekToEnd()
        guard size >= 22 else { throw RunnerError.failed("\(url.lastPathComponent) is not a ZIP archive.") }
        // The end-of-directory record may be followed by a comment of up to 65,535 bytes.
        let tail = try raw(size - UInt64(Swift.min(Int(size), 22 + 65_535)), Int(Swift.min(size, UInt64(22 + 65_535))))
        guard let at = stride(from: tail.count - 22, through: 0, by: -1).first(where: { tail.u32($0) == 0x0605_4b50 }) else {
            throw RunnerError.failed("\(url.lastPathComponent) is not a ZIP archive.")
        }
        let endOffset = size - UInt64(tail.count) + UInt64(at)
        let end = try raw(endOffset, 22)
        var count = Int(end.u16(10)), directorySize = UInt64(end.u32(12)), directoryOffset = UInt64(end.u32(16))
        if count == 0xFFFF || directorySize == 0xFFFF_FFFF || directoryOffset == 0xFFFF_FFFF {
            guard endOffset >= 20 else { throw RunnerError.failed("Truncated ZIP64 locator.") }
            let locator = try raw(endOffset - 20, 20)
            guard locator.u32(0) == 0x0706_4b50 else { throw RunnerError.failed("Missing ZIP64 locator.") }
            let record = try raw(locator.u64(8), 56)
            guard record.u32(0) == 0x0606_4b50, record.u64(32) <= 100_000 else { throw RunnerError.failed("Invalid ZIP64 directory record.") }
            count = Int(record.u64(32)); directorySize = record.u64(40); directoryOffset = record.u64(48)
        }
        guard count > 0, count <= 100_000, directorySize <= 32 << 20 else { throw RunnerError.failed("ZIP directory too large.") }
        let central = try raw(directoryOffset, Int(directorySize))
        var i = 0
        for _ in 0 ..< count {
            guard i + 46 <= central.count, central.u32(i) == 0x0201_4b50 else { throw RunnerError.failed("Invalid ZIP directory entry.") }
            guard central.u16(i + 10) == 0 else { throw RunnerError.failed("\(url.lastPathComponent): entries must be stored uncompressed.") }
            let n = Int(central.u16(i + 28)), x = Int(central.u16(i + 30)), c = Int(central.u16(i + 32))
            guard i + 46 + n + x + c <= central.count,
                let name = String(data: central.subdata(in: (central.startIndex + i + 46) ..< (central.startIndex + i + 46 + n)), encoding: .utf8),
                entries[name] == nil
            else { throw RunnerError.failed("Invalid ZIP entry name.") }
            var length = UInt64(central.u32(i + 24)), compressed = UInt64(central.u32(i + 20)), local = UInt64(central.u32(i + 42))
            if length == 0xFFFF_FFFF || compressed == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
                var j = i + 46 + n, found = false
                while j + 4 <= i + 46 + n + x {
                    let id = central.u16(j), extra = Int(central.u16(j + 2))
                    guard j + 4 + extra <= i + 46 + n + x else { break }
                    if id == 0x0001 {
                        var k = j + 4
                        let end = j + 4 + extra
                        /// The next 8-byte field, read only when it lies inside this extra field.
                        func field() throws -> UInt64 {
                            guard k + 8 <= end else { throw RunnerError.failed("Invalid ZIP64 extra field.") }
                            defer { k += 8 }
                            return central.u64(k)
                        }
                        if length == 0xFFFF_FFFF { length = try field() }
                        if compressed == 0xFFFF_FFFF { compressed = try field() }
                        if local == 0xFFFF_FFFF { local = try field() }
                        found = true
                    }
                    j += 4 + extra
                }
                guard found else { throw RunnerError.failed("Missing ZIP64 extra field.") }
            }
            guard length == compressed else { throw RunnerError.failed("\(url.lastPathComponent): entries must be stored uncompressed.") }
            let header = try raw(local, 30)
            guard header.u32(0) == 0x0403_4b50, header.u16(8) == 0 else { throw RunnerError.failed("Invalid ZIP local header.") }
            let payload = local + 30 + UInt64(header.u16(26)) + UInt64(header.u16(28))
            guard payload <= size, length <= size - payload, length <= UInt64(Int.max) else { throw RunnerError.failed("ZIP entry exceeds the archive.") }
            entries[name] = Entry(offset: payload, size: Int(length))
            i += 46 + n + x + c
        }
    }

    func size(of name: String) -> Int? { entries[name]?.size }

    func read(_ name: String, limit: Int) throws -> Data {
        guard let entry = entries[name] else { throw RunnerError.failed("Archive has no \(name).") }
        guard entry.size <= limit else { throw RunnerError.failed("\(name) is too large.") }
        return try raw(entry.offset, entry.size)
    }

    func read(_ name: String, from start: Int, count: Int) throws -> Data {
        guard let entry = entries[name], start >= 0, count >= 0, start <= entry.size, count <= entry.size - start else {
            throw RunnerError.failed("Invalid read of \(name).")
        }
        return try raw(entry.offset + UInt64(start), count)
    }

    private func raw(_ offset: UInt64, _ count: Int) throws -> Data {
        guard count >= 0, offset <= size, UInt64(count) <= size - offset else { throw RunnerError.failed("Invalid archive read.") }
        try file.seek(toOffset: offset)
        var data = Data(capacity: count)
        while data.count < count {
            guard let chunk = try file.read(upToCount: Swift.min(count - data.count, 64 << 20)), !chunk.isEmpty else {
                throw RunnerError.failed("Truncated archive.")
            }
            data.append(chunk)
        }
        return data
    }
}

private extension Data {
    func u16(_ i: Int) -> UInt16 { UInt16(self[startIndex + i]) | UInt16(self[startIndex + i + 1]) << 8 }
    func u32(_ i: Int) -> UInt32 { UInt32(u16(i)) | UInt32(u16(i + 2)) << 16 }
    func u64(_ i: Int) -> UInt64 { UInt64(u32(i)) | UInt64(u32(i + 4)) << 32 }
}
