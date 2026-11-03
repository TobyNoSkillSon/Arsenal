// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Suite struct ServingMemoryContractTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @Test(arguments: [
        "Workers/Chat/Sources", "Workers/Decisions/Sources", "Workers/Voices/Sources", "Workers/Sound/Sources",
        "models/d1-3b/runners/swift/Sources/D1Worker", "models/d1-3b/runners/swift/Sources/D1Engine",
        "models/flux2-klein-4b/runners/swift/Sources/FluxWorker", "models/flux2-klein-4b/runners/swift/Sources/Flux2Klein",
        "models/flux2-klein-4b/runners/swift/Sources/Flux3Klein",
        "models/birefnet/runners/swift/Sources/BiRefNetWorker", "models/birefnet/runners/swift/Sources/MLXBiRefNet",
        "Workers/Native/Sources", "Workers/Transport/Sources", "Sources/ArsenalCore/Runtime"
    ])
    func everyServingFamilyLeavesMemoryAdmissionToTheService(_ path: String) throws {
        let files = try #require(FileManager.default.enumerator(at: repo.appendingPathComponent(path), includingPropertiesForKeys: nil, options: .skipsHiddenFiles))
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
        try #require(!files.isEmpty, "No serving source audited at \(path)")
        // Cache limits recycle unused buffers; they are not live-model budgets. In contrast these mechanisms would
        // silently defeat Load Anyway even when the model otherwise loads and the wire remains responsive.
        let veto = #"vm\.swapusage|\b(?:memoryLimit|setMemoryLimit|set_memory_limit)\b|\b(?:setrlimit|RLIMIT_AS|RLIMIT_RSS)\b|makeMemoryPressureSource|\bhost_statistics64\b"#
        for file in files {
            let source = try String(contentsOf: file, encoding: .utf8)
            #expect(source.range(of: veto, options: .regularExpression) == nil, "Serving memory veto in \(file.path)")
        }
    }

    @Test func isolatedIntakeAndCheckersKeepTheirProtectiveBudgets() throws {
        let intake = try String(contentsOf: repo.appendingPathComponent("Sources/ArsenalCore/Measurement/ModelIntakeCommand.swift"), encoding: .utf8)
        #expect(intake.contains("swap > initialSwap") && intake.contains("bytes > limit") && intake.contains("kill(pid, SIGKILL)"))
        let admission = try String(contentsOf: repo.appendingPathComponent("Sources/ArsenalCore/Measurement/MeasurementMemoryAdmission.swift"), encoding: .utf8)
        #expect(admission.contains("reserveMemory(") && admission.contains("waitForMemory: false"))
        let checker = try String(contentsOf: repo.appendingPathComponent("models/birefnet/runners/swift/Sources/birefnet-check/main.swift"), encoding: .utf8)
        #expect(checker.contains("Memory.memoryLimit = 12 * 1024 * 1024 * 1024"))
    }
}
