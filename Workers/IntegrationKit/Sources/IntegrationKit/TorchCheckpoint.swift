// SPDX-License-Identifier: MIT
// The one PyTorch checkpoint reader for every native family (Audio, Sound, Restoration); families refer to it through
// a module-local typealias.
import Foundation
import MLX
import ModelPrecision

/// Bounded, non-executing reader for PyTorch `torch.save` ZIP checkpoints (stored entries, classic or ZIP64 directory).
/// `data.pkl` is walked opcode by opcode: globals are recorded by name only, nothing is imported or called. Only the
/// reductions a state dict needs are interpreted (`OrderedDict()`, `_rebuild_tensor_v2`, `_rebuild_parameter`);
/// any other call becomes an opaque value. Get has verified the file's SHA-256 before a converter opens it.
public final class TorchCheckpoint {
    public struct Tensor: Equatable {
        public let storage: String
        public let dtype: DType
        public let offset: Int
        public let shape: [Int]
        public let stride: [Int]
        public init(storage: String, dtype: DType, offset: Int, shape: [Int], stride: [Int]) {
            self.storage = storage; self.dtype = dtype; self.offset = offset; self.shape = shape; self.stride = stride
        }
    }
    private struct Entry { let offset: UInt64; let size: Int }
    private let file: FileHandle
    private let size: UInt64
    /// Byte offset of the ZIP inside `file` (a checkpoint stored in a container such as a `.nemo` tar); 0 for a plain file.
    private let base: UInt64
    private let entries: [String: Entry]
    private let prefix: String
    /// The unpickled root object.
    public let root: TorchPickle.Value

    deinit { try? file.close() }
    public convenience init(_ url: URL) throws {
        let file = try FileHandle(forReadingFrom: url)
        try self.init(file, base: 0, size: try file.seekToEnd())
    }

    /// The checkpoint occupying bytes `base ..< base + size` of `file` (stored uncompressed in an outer archive).
    public init(_ file: FileHandle, base: UInt64, size: UInt64) throws {
        let end = try file.seekToEnd()
        guard base <= end, size <= end - base else { throw PrecisionError("Checkpoint range exceeds its file") }
        self.file = file; self.size = size; self.base = base
        entries = try Self.directory(file, base: base, size: size)
        guard let pickle = entries.keys.filter({ $0.hasSuffix("/data.pkl") || $0 == "data.pkl" }).sorted().first else {
            throw PrecisionError("Checkpoint has no data.pkl")
        }
        prefix = String(pickle.dropLast("data.pkl".count))
        guard let entry = entries[pickle], entry.size <= 64 << 20 else { throw PrecisionError("Checkpoint pickle is too large") }
        root = try TorchPickle.parse(try Self.read(file, base: base, size: size, entry.offset, entry.size))
    }

    /// Every tensor of the state dict at `path` (empty: the root object), keyed by its name.
    public func tensors(at path: [String] = []) throws -> [String: Tensor] {
        var value = root
        for key in path {
            guard case .dict(let box) = value, let next = box.items.first(where: { $0.0 == .string(key) })?.1 else {
                throw PrecisionError("Checkpoint has no " + path.joined(separator: "."))
            }
            value = next
        }
        guard case .dict(let box) = value else { throw PrecisionError("Checkpoint state dict is not a dictionary") }
        var out: [String: Tensor] = [:]
        for (key, item) in box.items {
            guard case .string(let name) = key else { continue }
            if case .tensor(let tensor) = item { out[name] = tensor }
        }
        return out
    }

    /// Reads one contiguous tensor as an MLX array of its stored dtype.
    public func array(_ tensor: Tensor) throws -> MLXArray {
        guard let entry = entries[prefix + "data/" + tensor.storage] else { throw PrecisionError("Missing tensor storage " + tensor.storage) }
        var count = 1
        for dim in tensor.shape {
            let (product, overflow) = count.multipliedReportingOverflow(by: dim)
            // MLX dimensions are 32-bit; a larger one cannot be a real tensor even when another dimension is 0.
            guard dim >= 0, dim <= Int(Int32.max), !overflow else { throw PrecisionError("Invalid tensor shape") }
            count = product
        }
        var contiguous = 1
        for (dim, stride) in zip(tensor.shape, tensor.stride).reversed() where dim > 1 {
            guard stride == contiguous else { throw PrecisionError("Non-contiguous checkpoint tensor") }
            let (product, overflow) = contiguous.multipliedReportingOverflow(by: dim)
            guard !overflow else { throw PrecisionError("Invalid tensor shape") }
            contiguous = product
        }
        let width = tensor.dtype.size
        let (start, startOverflow) = tensor.offset.multipliedReportingOverflow(by: width)
        let (bytes, bytesOverflow) = count.multipliedReportingOverflow(by: width)
        guard tensor.offset >= 0, !startOverflow, !bytesOverflow, start <= entry.size, bytes <= entry.size - start else { throw PrecisionError("Tensor exceeds its storage") }
        let data = try read(entry.offset + UInt64(start), bytes)
        switch tensor.dtype {
        case .float32: return MLXArray(data, tensor.shape, type: Float.self)
        case .float16: return MLXArray(data, tensor.shape, type: Float16.self)
        case .int64: return MLXArray(data, tensor.shape, type: Int64.self)
        case .int32: return MLXArray(data, tensor.shape, type: Int32.self)
        default: throw PrecisionError("Unsupported checkpoint dtype")
        }
    }

    private func read(_ offset: UInt64, _ count: Int) throws -> Data { try Self.read(file, base: base, size: size, offset, count) }
    private static func read(_ file: FileHandle, base: UInt64, size: UInt64, _ offset: UInt64, _ count: Int) throws -> Data {
        guard count >= 0, count <= 256 << 20, offset <= size, UInt64(count) <= size - offset else { throw PrecisionError("Invalid checkpoint read") }
        try file.seek(toOffset: base + offset)
        let data = try file.read(upToCount: count) ?? Data()
        guard data.count == count else { throw PrecisionError("Truncated checkpoint") }
        return data
    }

    private static func directory(_ file: FileHandle, base: UInt64, size: UInt64) throws -> [String: Entry] {
        func read(_ offset: UInt64, _ count: Int) throws -> Data { try Self.read(file, base: base, size: size, offset, count) }
        var entries: [String: Entry] = [:]
        guard size >= 22 else { throw PrecisionError("Truncated checkpoint archive") }
        let end = try read(size - 22, 22)
        guard end.u32(0) == 0x0605_4b50, end.u16(20) == 0 else { throw PrecisionError("Unsupported checkpoint archive footer") }
        var count = Int(end.u16(10)), directorySize = UInt64(end.u32(12)), directoryOffset = UInt64(end.u32(16))
        if count == 0xFFFF || directorySize == 0xFFFF_FFFF || directoryOffset == 0xFFFF_FFFF {
            guard size >= 42 else { throw PrecisionError("Truncated ZIP64 locator") }
            let locator = try read(size - 42, 20)
            guard locator.u32(0) == 0x0706_4b50 else { throw PrecisionError("Missing ZIP64 locator") }
            let record = try read(locator.u64(8), 56)
            guard record.u32(0) == 0x0606_4b50 else { throw PrecisionError("Invalid ZIP64 directory record") }
            guard record.u64(32) <= 100_000 else { throw PrecisionError("Checkpoint directory too large") }
            count = Int(record.u64(32)); directorySize = record.u64(40); directoryOffset = record.u64(48)
        }
        guard count > 0, count <= 100_000, directorySize <= 32 << 20 else { throw PrecisionError("Checkpoint directory too large") }
        let central = try read(directoryOffset, Int(directorySize))
        var i = 0
        for _ in 0..<count {
            guard i + 46 <= central.count, central.u32(i) == 0x0201_4b50, central.u16(i + 10) == 0 else {
                throw PrecisionError("Checkpoint entries must be stored uncompressed")
            }
            let n = Int(central.u16(i + 28)), x = Int(central.u16(i + 30)), c = Int(central.u16(i + 32))
            guard i + 46 + n + x + c <= central.count, let name = String(data: central.subdata(in: (i + 46)..<(i + 46 + n)), encoding: .utf8),
                entries[name] == nil
            else { throw PrecisionError("Invalid checkpoint entry") }
            var length = UInt64(central.u32(i + 24)), compressed = UInt64(central.u32(i + 20)), local = UInt64(central.u32(i + 42))
            if length == 0xFFFF_FFFF || compressed == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
                var j = i + 46 + n, found = false
                while j + 4 <= i + 46 + n + x {
                    let id = central.u16(j), extra = Int(central.u16(j + 2))
                    guard j + 4 + extra <= i + 46 + n + x else { break }
                    if id == 0x0001 {
                        // Each 8-byte value is bounds-checked before it is read: a short field must not read past the directory.
                        var k = j + 4
                        func value() throws -> UInt64 {
                            guard k + 8 <= j + 4 + extra else { throw PrecisionError("Invalid ZIP64 extra field") }
                            defer { k += 8 }
                            return central.u64(k)
                        }
                        if length == 0xFFFF_FFFF { length = try value() }
                        if compressed == 0xFFFF_FFFF { compressed = try value() }
                        if local == 0xFFFF_FFFF { local = try value() }
                        found = true
                    }
                    j += 4 + extra
                }
                guard found else { throw PrecisionError("Missing ZIP64 extra field") }
            }
            guard length == compressed else { throw PrecisionError("Checkpoint entries must be stored uncompressed") }
            let header = try read(local, 30)
            guard header.u32(0) == 0x0403_4b50, header.u16(8) == 0 else { throw PrecisionError("Invalid checkpoint local header") }
            let payload = local + 30 + UInt64(header.u16(26)) + UInt64(header.u16(28))
            guard payload <= size, length <= size - payload, length <= UInt64(Int.max) else { throw PrecisionError("Checkpoint entry exceeds archive") }
            entries[name] = Entry(offset: payload, size: Int(length))
            i += 46 + n + x + c
        }
        return entries
    }
}

/// The pickle subset `torch.save` writes for state dicts (protocol 2). Values that need code to construct are kept as
/// named placeholders and never executed.
public enum TorchPickle {
    public final class DictBox {
        public var items: [(Value, Value)] = []
        /// Position of each string key, so a large state dict builds in linear time.
        public var strings: [String: Int] = [:]
        /// Position of each other scalar key (int, float, bool, bytes, None), for the same reason.
        var scalars: [ScalarKey: Int] = [:]
        /// Nesting below this dict, and whether it already sits inside another value (then it no longer changes).
        var depth = 1, sealed = false
    }
    /// A scalar dict key as `Value ==` compares it (an int and a bool never match). A float is keyed on its bits, with
    /// -0.0 as 0.0, so a NaN key (a memoized NaN fetched again, which Python's dict matches by identity) replaces its
    /// entry in one lookup instead of landing in one hash bucket with every earlier NaN, which built quadratically.
    enum ScalarKey: Hashable {
        case none, bool(Bool), int(Int), float(UInt64), bytes(Data)
        init?(_ value: Value) {
            switch value {
            case .none: self = .none
            case .bool(let v): self = .bool(v)
            case .int(let v): self = .int(v)
            case .float(let v): self = .float(v == 0 ? 0 : v.bitPattern)
            case .bytes(let v): self = .bytes(v)
            default: return nil
            }
        }
    }
    public final class ListBox {
        public var items: [Value] = []
        var depth = 1, sealed = false
    }
    public indirect enum Value: Equatable {
        case none, bool(Bool), int(Int), float(Double), string(String), bytes(Data)
        case tuple([Value]), list(ListBox), dict(DictBox)
        case global(String), storage(key: String, dtype: DType), tensor(TorchCheckpoint.Tensor), opaque(String), mark
        public static func == (lhs: Value, rhs: Value) -> Bool {
            switch (lhs, rhs) {
            case (.none, .none), (.mark, .mark): return true
            case (.bool(let a), .bool(let b)): return a == b
            case (.int(let a), .int(let b)): return a == b
            case (.float(let a), .float(let b)): return a == b
            case (.string(let a), .string(let b)), (.global(let a), .global(let b)), (.opaque(let a), .opaque(let b)): return a == b
            case (.bytes(let a), .bytes(let b)): return a == b
            case (.tuple(let a), .tuple(let b)): return a == b
            case (.list(let a), .list(let b)): return a === b
            case (.dict(let a), .dict(let b)): return a === b
            case (.storage(let a, let x), .storage(let b, let y)): return a == b && x == y
            case (.tensor(let a), .tensor(let b)): return a == b
            default: return false
            }
        }
    }
    /// Longest string or bytes dict key accepted.
    static let maximumKeyBytes = 65_536
    static let storageTypes: [String: DType] = [
        "torch.FloatStorage": .float32, "torch.HalfStorage": .float16, "torch.BFloat16Storage": .bfloat16,
        "torch.LongStorage": .int64, "torch.IntStorage": .int32,
        // Parsed so a state dict holding complex buffers (SAM 3.1's RoPE `freqs_cis`) can be read; `array` refuses them.
        "torch.ComplexFloatStorage": .complex64
    ]

    /// Lists and dicts are filled while they are on the stack, so their depth is read live from the box.
    private static func depth(_ entry: (value: Value, depth: Int)) -> Int {
        switch entry.value {
        case .list(let box): return box.depth
        case .dict(let box): return box.depth
        default: return entry.depth
        }
    }
    /// A value placed inside another is final: a later APPEND or SETITEM into it could deepen its parents unseen.
    private static func seal(_ entries: [(value: Value, depth: Int)]) -> Int {
        var deepest = 0
        for entry in entries {
            if case .list(let box) = entry.value { box.sealed = true }
            if case .dict(let box) = entry.value { box.sealed = true }
            deepest = max(deepest, depth(entry))
        }
        return deepest
    }
    private static func reduce(_ callable: Value, _ arguments: Value) throws -> Value {
        func fail(_ message: String) -> PrecisionError { PrecisionError("Checkpoint pickle: " + message) }
        guard case .global(let name) = callable, case .tuple(let args) = arguments else { return .opaque("call") }
        switch name {
        case "collections.OrderedDict": return .dict(DictBox())
        case "torch._utils._rebuild_parameter":
            return args.first ?? .none
        case "torch._utils._rebuild_tensor_v2":
            guard args.count >= 4, case .storage(let key, let dtype) = args[0], case .int(let offset) = args[1],
                case .tuple(let size) = args[2], case .tuple(let stride) = args[3]
            else { throw fail("malformed tensor") }
            let shape = try size.map { (value: Value) -> Int in
                guard case .int(let v) = value else { throw fail("shape") }; return v
            }
            let strides = try stride.map { (value: Value) -> Int in
                guard case .int(let v) = value else { throw fail("stride") }; return v
            }
            guard shape.count == strides.count else { throw fail("rank") }
            return .tensor(TorchCheckpoint.Tensor(storage: key, dtype: dtype, offset: offset, shape: shape, stride: strides))
        default: return .opaque(name)
        }
    }
    public static func parse(_ data: Data) throws -> Value {
        let bytes = [UInt8](data)
        // Every value carries its nesting depth, worked out once from its parts, so a tuple built from many references
        // to one shared value costs no more to check than its own items (a recursive walk would be exponential).
        var i = 0, stack: [(value: Value, depth: Int)] = [], memo: [Int: (value: Value, depth: Int)] = [:], pushed = 0
        func fail(_ message: String) -> PrecisionError { PrecisionError("Checkpoint pickle: " + message) }
        /// Bounds memory (values created) and nesting (recursive comparison and release) for a hostile file.
        func push(_ value: Value, depth: Int = 0) throws {
            pushed += 1
            guard pushed <= 8_000_000 else { throw fail("too many values") }
            guard depth <= 32 else { throw fail("nesting too deep") }
            stack.append((value, depth))
        }
        func remember(_ key: Int) throws {
            guard memo.count < 1_000_000 || memo[key] != nil else { throw fail("too many values") }
            guard let top = stack.last else { throw fail("memo of an empty stack") }
            memo[key] = top
        }
        func tuple(_ items: [(value: Value, depth: Int)]) throws { try push(.tuple(items.map(\.value)), depth: 1 + Self.seal(items)) }
        func take(_ n: Int) throws -> ArraySlice<UInt8> {
            guard n >= 0, i + n <= bytes.count else { throw fail("truncated") }
            defer { i += n }
            return bytes[i..<(i + n)]
        }
        func uint(_ n: Int) throws -> Int { try take(n).reversed().reduce(0) { $0 << 8 | Int($1) } }
        func popEntry() throws -> (value: Value, depth: Int) { guard let entry = stack.popLast() else { throw fail("stack underflow") }; return entry }
        func pop() throws -> Value { try popEntry().value }
        func popMark() throws -> [(value: Value, depth: Int)] {
            guard let index = stack.lastIndex(where: { $0.value == .mark }) else { throw fail("missing mark") }
            let items = Array(stack[(index + 1)...]); stack.removeSubrange(index...)
            return items
        }
        func line() throws -> String {
            guard let end = bytes[i...].firstIndex(of: 0x0A) else { throw fail("unterminated line") }
            defer { i = end + 1 }
            return String(decoding: bytes[i..<end], as: UTF8.self)
        }
        func setItems(_ items: [(value: Value, depth: Int)], into target: Value) throws {
            guard items.count % 2 == 0, case .dict(let box) = target, !box.sealed else { throw fail("SETITEMS needs an open dict") }
            box.depth = max(box.depth, 1 + Self.seal(items))
            guard box.depth <= 32 else { throw fail("nesting too deep") }
            for index in stride(from: 0, to: items.count, by: 2) {
                let key = items[index].value
                // Every lookup hashes the key, so one huge key fetched back from the memo many times would cost its size
                // each time; no checkpoint names a tensor with 64 KB.
                switch key {
                case .string(let name) where name.utf8.count > Self.maximumKeyBytes: throw fail("dict key over 64 KB")
                case .bytes(let data) where data.count > Self.maximumKeyBytes: throw fail("dict key over 64 KB")
                default: break
                }
                let existing: Int?
                let scalar = ScalarKey(key)
                // Strings and other scalars are indexed; a container key is kept as given (state dicts have none).
                if case .string(let name) = key { existing = box.strings[name] } else { existing = scalar.flatMap { box.scalars[$0] } }
                if let existing {
                    box.items[existing].1 = items[index + 1].value
                } else {
                    if case .string(let name) = key { box.strings[name] = box.items.count } else if let scalar { box.scalars[scalar] = box.items.count }
                    box.items.append((key, items[index + 1].value))
                }
            }
        }
        func append(_ items: [(value: Value, depth: Int)]) throws {
            guard case .list(let box)? = stack.last?.value, !box.sealed else { throw fail("APPEND needs an open list") }
            // Appended before sealing, so a list may hold itself (a recursive pickle) without refusing its own append.
            box.items.append(contentsOf: items.map(\.value))
            box.depth = max(box.depth, 1 + Self.seal(items))
            guard box.depth <= 32 else { throw fail("nesting too deep") }
        }
        while true {
            guard i < bytes.count else { throw fail("missing STOP") }
            let op = bytes[i]; i += 1
            switch op {
            case 0x80: _ = try take(1) // PROTO
            case 0x95: _ = try take(8) // FRAME
            case 0x2E: return try pop() // STOP
            case 0x28: try push(.mark) // MARK
            case 0x7D: try push(.dict(DictBox())) // EMPTY_DICT
            case 0x5D: try push(.list(ListBox())) // EMPTY_LIST
            case 0x29: try push(.tuple([])) // EMPTY_TUPLE
            case 0x4E: try push(.none) // NONE
            case 0x88: try push(.bool(true)) // NEWTRUE
            case 0x89: try push(.bool(false)) // NEWFALSE
            case 0x4B: try push(.int(try uint(1))) // BININT1
            case 0x4D: try push(.int(try uint(2))) // BININT2
            case 0x4A: // BININT (signed 32-bit)
                try push(.int(Int(Int32(truncatingIfNeeded: UInt32(try uint(4))))))
            case 0x8A: // LONG1
                let n = try uint(1)
                guard n <= 8 else { throw fail("long too wide") }
                let raw = try take(n)
                var value = 0
                for (k, byte) in raw.enumerated() { value |= Int(byte) << (8 * k) }
                if n > 0, n < 8, raw.last! & 0x80 != 0 { value -= 1 << (8 * n) }
                try push(.int(value))
            case 0x47: // BINFLOAT (big-endian, as pickle writes it)
                let raw = try take(8).reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
                try push(.float(Double(bitPattern: raw)))
            case 0x58: try push(.string(String(decoding: try take(try uint(4)), as: UTF8.self))) // BINUNICODE
            case 0x8C: try push(.string(String(decoding: try take(try uint(1)), as: UTF8.self))) // SHORT_BINUNICODE
            case 0x43: try push(.bytes(Data(try take(try uint(1))))) // SHORT_BINBYTES
            case 0x42: try push(.bytes(Data(try take(try uint(4))))) // BINBYTES
            case 0x55: try push(.string(String(decoding: try take(try uint(1)), as: UTF8.self))) // SHORT_BINSTRING
            case 0x54: try push(.string(String(decoding: try take(try uint(4)), as: UTF8.self))) // BINSTRING
            case 0x63: // GLOBAL module\nname\n
                let module = try line(), name = try line()
                try push(.global(module + "." + name))
            case 0x93: // STACK_GLOBAL
                let name = try pop(), module = try pop()
                guard case .string(let n) = name, case .string(let m) = module else { throw fail("STACK_GLOBAL") }
                try push(.global(m + "." + n))
            case 0x71: try remember(try uint(1)) // BINPUT
            case 0x72: try remember(try uint(4)) // LONG_BINPUT
            case 0x94: try remember(memo.count) // MEMOIZE
            case 0x68: // BINGET
                guard let entry = memo[try uint(1)] else { throw fail("memo") }
                try push(entry.value, depth: entry.depth)
            case 0x6A: // LONG_BINGET
                guard let entry = memo[try uint(4)] else { throw fail("memo") }
                try push(entry.value, depth: entry.depth)
            case 0x74: try tuple(try popMark()) // TUPLE
            case 0x85: try tuple([try popEntry()]) // TUPLE1
            case 0x86: // TUPLE2
                let b = try popEntry(), a = try popEntry(); try tuple([a, b])
            case 0x87: // TUPLE3
                let c = try popEntry(), b = try popEntry(), a = try popEntry(); try tuple([a, b, c])
            case 0x52: // REDUCE
                let arguments = try popEntry(), callable = try pop()
                // `_rebuild_parameter` hands back one of its arguments, so the result is at most one level shallower.
                try push(try Self.reduce(callable, arguments.value), depth: max(0, Self.depth(arguments) - 1))
            case 0x51: // BINPERSID
                guard case .tuple(let id) = try pop(), id.count >= 3, id[0] == .string("storage"), case .global(let type) = id[1],
                    case .string(let key) = id[2], let dtype = storageTypes[type]
                else { throw fail("unsupported persistent id") }
                try push(.storage(key: key, dtype: dtype))
            case 0x81: // NEWOBJ (cls.__new__(cls, *args), e.g. a Lightning checkpoint's dataclasses): kept opaque, never constructed
                _ = try pop()
                guard case .global(let name) = try pop() else { try push(.opaque("object")); break }
                try push(.opaque(name))
            case 0x62: _ = try pop() // BUILD: the state (OrderedDict `_metadata`) is not needed
            case 0x73: // SETITEM
                let value = try popEntry(), key = try popEntry()
                guard let target = stack.last?.value else { throw fail("SETITEM") }
                try setItems([key, value], into: target)
            case 0x75: // SETITEMS
                let items = try popMark()
                guard let target = stack.last?.value else { throw fail("SETITEMS") }
                try setItems(items, into: target)
            case 0x61: try append([try popEntry()]) // APPEND
            case 0x65: try append(try popMark()) // APPENDS
            default: throw fail(String(format: "unsupported opcode 0x%02x", op))
            }
        }
    }
}

private extension Data {
    func u16(_ i: Int) -> UInt16 { UInt16(self[startIndex + i]) | UInt16(self[startIndex + i + 1]) << 8 }
    func u32(_ i: Int) -> UInt32 { UInt32(u16(i)) | UInt32(u16(i + 2)) << 16 }
    func u64(_ i: Int) -> UInt64 { UInt64(u32(i)) | UInt64(u32(i + 4)) << 32 }
}
