// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct ToolNamesManifestTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @Test func capabilityWindowNamesNeedOneOfTheirOperations() throws {
        let folder = repo.appendingPathComponent("models/da3-small")
        var model = try ModelManifest.load(folder)
        #expect(model.operations.contains { $0.name == "estimate_depth" })
        func toolProblems(_ tools: [String]) -> [String] { model.tools = tools; return model.problems(folder: folder).filter { $0.contains("tools") } }
        #expect(toolProblems(["operation", "depth"]).isEmpty)
        #expect(toolProblems(["find"]).contains { $0.contains("no declared operation") })
        #expect(toolProblems(["depth", "depth"]).contains { $0.contains("unique") })
        #expect(toolProblems(["holograms"]).contains { $0.contains("unique") })
    }
}
