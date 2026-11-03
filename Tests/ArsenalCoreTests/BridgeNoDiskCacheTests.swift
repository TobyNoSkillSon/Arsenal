// SPDX-License-Identifier: MIT
import Foundation
import Testing

/// VM run, 2 Nov: the stdio bridge used URLSession.shared, whose disk cache and cookie store outlived an uninstall
/// (~/Library/Caches/arsenal-mcp, ~/Library/HTTPStorages/arsenal-mcp). It talks only to the local app, so it keeps none.
@Test func stdioBridgeUsesOnlyAnEphemeralSession() throws {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let sources = try FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("Sources/arsenal-mcp"), includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "swift" }
    let text = try sources.map { try String(contentsOf: $0, encoding: .utf8) }.joined()
    #expect(!sources.isEmpty && !text.contains("URLSession.shared."))
    #expect(text.contains("URLSession(configuration: .ephemeral)"))
}
