// SPDX-License-Identifier: MIT
import Foundation
// Retain the upstream name at this internal port boundary, never the upstream data directory.
public enum VireoFiles {
    public static var support: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"]
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Arsenal").path)
    }
}
