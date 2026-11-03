// SPDX-License-Identifier: MIT
// Loading holds the weights once. Each tensor's file bytes used to stay alive (autoreleased) until a whole component
// had loaded, so the real snapshot peaked at 37.7 GB for 20 GB of weights (1 Nov 2026); now 19.8 GB.
import Darwin
import Foundation
import MLX
import Testing
@testable import Flux3Klein

@Suite(.serialized) struct Flux3LoadMemory {
    static func footprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return status == KERN_SUCCESS ? Int64(info.phys_footprint) : -1
    }

    /// `tensors` BF16 tensors of `bytes` each in one safetensors file.
    static func archive(in directory: URL, tensors: Int, bytes: Int) throws {
        var header: [String: Any] = [:]
        for i in 0..<tensors { header[String(format: "w%03d", i)] = ["dtype": "BF16", "shape": [bytes / 2], "data_offsets": [i * bytes, (i + 1) * bytes]] }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var file = Data(); var length = UInt64(json.count).littleEndian
        file.append(Data(bytes: &length, count: 8)); file.append(json)
        let payload = Data((0..<bytes).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ 7) })
        for _ in 0..<tensors { file.append(payload) }
        try file.write(to: directory.appendingPathComponent("w.safetensors"))
    }

    @Test func loadingKeepsOneCopyOfTheWeights() throws {
        useTestMetallib()
        MLX.GPU.clearCache()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("flux3-load-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tensors = 96, bytes = 4 << 20, total = Int64(tensors * bytes)
        try Self.archive(in: directory, tensors: tensors, bytes: bytes)
        // The pool around the call stands for the worker's run loop: whatever the load leaves autoreleased is still
        // alive when the call returns, which is where the doubled peak showed.
        try autoreleasepool {
            let before = Self.footprint()
            let weights = try Flux3Weights.tensors(directory, dtype: .bfloat16, name: { $0 })
            let grown = Self.footprint() - before
            #expect(weights.count == tensors)
            #expect(before > 0 && grown < total * 3 / 2, "load grew the footprint by \(grown) bytes for \(total) bytes of weights")
        }
    }
}
