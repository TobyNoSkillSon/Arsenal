// SPDX-License-Identifier: MIT

/// The MCP server's name and its tool-name prefix: every tool name, description, hint and test derives from these two,
/// so renaming the product changes them in one place.
public enum MCPNames {
    public static let server = "Arsenal"
    public static let prefix = "arsenal_"
    /// A tool's full name: `tool("run")` is "arsenal_run".
    public static func tool(_ name: String) -> String { prefix + name }
}
