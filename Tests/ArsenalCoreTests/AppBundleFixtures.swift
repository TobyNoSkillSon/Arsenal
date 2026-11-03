// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// Unregister before deleting; include renamed backups and any already-removed registered paths.
func unregisterFixtureBundles(in root: URL) {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    do {
        try UpdateVerification.tool("/usr/bin/python3", ["-B", repo.appendingPathComponent("scripts/launch_services.py").path, "cleanup-tree", root.path], timeout: 60)
    } catch {
        Issue.record("Could not unregister fixture bundles: \(error)")
    }
}
