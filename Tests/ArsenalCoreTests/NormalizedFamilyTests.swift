// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The native families whose weights are normalized at Get are listed in three places: the app (NativeHalfPolicy), the
/// lab's dtype rule (scripts/model-check.py) and the shipped native family set (scripts/pkg-lib.sh). They must agree,
/// and a [precision] block belongs only to these families.
struct NormalizedFamilyTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @Test func normalizedFamiliesAgreeAcrossAppAndScripts() throws {
        let python = try String(contentsOf: repo.appendingPathComponent("scripts/model-check.py"), encoding: .utf8)
        let marker = try #require(python.range(of: "native = any(r.get('worker') in ("))
        let pythonFamilies = Set(python[marker.upperBound...].prefix { $0 != ")" }.split(separator: ",").map { $0.trimmingCharacters(in: CharacterSet(charactersIn: " '")) })
        let shell = try String(contentsOf: repo.appendingPathComponent("scripts/pkg-lib.sh"), encoding: .utf8)
        let open = try #require(shell.range(of: "NATIVE_FAMILIES=("))
        let shipped = Set(shell[open.upperBound...].prefix { $0 != ")" }.split(separator: " ").map(String.init))
        let app = Set(NativeHalfPolicy.normalizedWorkers)
        #expect(app == pythonFamilies, "scripts/model-check.py base_precision")
        #expect(app.isSubset(of: shipped), "scripts/pkg-lib.sh NATIVE_FAMILIES")
    }

    @Test func precisionBlockIsRefusedOutsideNormalizedFamilies() throws {
        let folder = repo.appendingPathComponent("models/big-lama")
        var model = try ModelManifest.load(folder)
        #expect(!model.problems(folder: folder).contains { $0.contains("[precision]") })
        model.runner = model.runner.map { runner in
            var runner = runner; runner.worker = "chat"; return runner
        }
        #expect(model.problems(folder: folder).contains { $0.contains("[precision] is only for the normalized native families") })
    }
}
