// SPDX-License-Identifier: MIT
import Foundation

/// The service captures this at admission; child tasks inherit it, detached tasks do not.
public enum ActivityRequestContext {
    @TaskLocal public static var requester: String = "Arsenal"

    public static func identity(headers: [String: String]) -> String {
        let header = headers["x-arsenal-client"] ?? headers["x-mobius-client"] // legacy-compat: clients written before the rename
        if let name = header?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty {
            return displayName(String(name.prefix(128)))
        }
        // A User-Agent names a library, not who is asking (no guessing); the panel letters unnamed streams.
        return "API"
    }

    /// One spelling per client, wherever its name is shown (activity, playback): a known MCP `clientInfo.name` slug
    /// becomes its product's name, and any other all-lowercase slug ("my-agent") reads as words ("My Agent"). Names a
    /// client already spells for people are kept.
    public static func displayName(_ raw: String) -> String {
        let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let known = knownClients[name.lowercased()] { return known }
        guard !name.isEmpty, name == name.lowercased(), name.contains(where: { $0 == "-" || $0 == "_" }), !name.contains(" ") else { return name }
        return name.split(whereSeparator: { $0 == "-" || $0 == "_" }).map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined(separator: " ")
    }
    static let knownClients: [String: String] = [
        "claude-code": "Claude Code", "claude code": "Claude Code", "claude-ai": "Claude", "codex": "Codex", "codex-mcp-client": "Codex",
        "codex_cli": "Codex", "cursor": "Cursor", "cursor-vscode": "Cursor", "vscode": "VS Code", "visual-studio-code": "VS Code", "gemini-cli": "Gemini CLI",
        "windsurf": "Windsurf", "zed": "Zed", "t3-code": "T3 Code", "mcp-inspector": "MCP Inspector"
    ]
}
