// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

/// A pending revision or a missing byte keeps a Get due even while the previous revision is loaded.
@Test func mcpDownloadHonorsAPendingPlanWhilePreviousRevisionIsLoaded() async throws {
    for (state, bytes): (ModelState, Int64) in [(.ready, 1), (.loaded, 1), (.loaded, 0)] {
        var row = ModelRow(id: "pending", name: "Pending revision", kind: "voices", states: ["bf16": .init(state: state)], downloadBytes: ["bf16": bytes])
        if bytes == 0 { row.revision = .init(active: "previous", pending: "approved-candidate") }
        let h = try await MCPHarness(rows: [row])
        let answer = try await h.text("download", ["model": "pending"])
        if bytes == 1 {
            #expect(answer.hasPrefix("Not downloaded: Pending revision isn't downloaded yet — 1 byte,"), "\(answer)")
            #expect(await h.service.getCalls.isEmpty)
            let started = try await h.text("download", ["model": "pending", "allow_download": true])
            #expect(!started.hasPrefix("Not downloaded"), "\(started)")
        } else {
            #expect(!answer.hasPrefix("Not downloaded") && !answer.contains("already downloaded"), "\(answer)")
        }
        if let job = await h.router.mcpJobs.keys.first {
            #expect(!(try await h.text("wait", ["job": .string(job), "timeout_s": 600])).hasPrefix("["))
        }
        #expect(await h.service.getCalls == ["pending bf16"])
        await h.close()
    }
}
