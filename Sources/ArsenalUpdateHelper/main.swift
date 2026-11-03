// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

@main
struct ArsenalUpdateHelper {
    @MainActor static func main() async {
        let arguments = ProcessInfo.processInfo.arguments
        guard arguments.count == 3, arguments[1] == "--plan" else {
            FileHandle.standardError.write(Data("Usage: ArsenalUpdateHelper --plan <plan.json>\n".utf8)); exit(64)
        }
        do {
            let plan = try JSONDecoder().decode(UpdateInstallPlan.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[2])))
            try await UpdateInstaller.install(plan)
        } catch {
            FileHandle.standardError.write(Data("Arsenal update failed: \(error.localizedDescription)\n".utf8)); exit(1)
        }
    }
}
