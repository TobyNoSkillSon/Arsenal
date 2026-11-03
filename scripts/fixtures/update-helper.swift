// SPDX-License-Identifier: MIT
// Same installer code, with constructor-only short deadlines and SIGSTOP checkpoints for failure testing.
import AppKit
import Darwin
import Foundation

@main struct UpdateFixtureHelper {
    @MainActor static func main() async {
        let args = CommandLine.arguments
        if args.count == 3, args[1] == "--reconcile" {
            UpdateInstaller.reconcile(URL(fileURLWithPath: args[2])); exit(0)
        }
        guard args.count == 3, args[1] == "--plan", let rootPath = ProcessInfo.processInfo.environment["ARSENAL_UPDATE_TEST_ROOT"] else { exit(64) }
        let root = URL(fileURLWithPath: rootPath)
        let mode = ProcessInfo.processInfo.environment["ARSENAL_UPDATE_TEST_MODE"] ?? "good"
        do {
            let plan = try JSONDecoder().decode(UpdateInstallPlan.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
            try Data(plan.staged.directory.path.utf8).write(to: root.appendingPathComponent("stage-dir"), options: .atomic)
            try Data("\(getpid())".utf8).write(to: root.appendingPathComponent("helper-pid"), options: .atomic)
            var limits = UpdateInstaller.Limits()
            limits.exit = 5; limits.lateExit = 5; limits.launch = 4; limits.restore = ["slow-startup", "restore-slow"].contains(mode) ? 15 : 4
            let opener: UpdateInstaller.Open = { app, config, completion in
                let counter = root.appendingPathComponent("open-count")
                let value = ((try? String(contentsOf: counter, encoding: .utf8)).flatMap(Int.init) ?? 0) + 1
                try? Data("\(value)".utf8).write(to: counter, options: .atomic)
                if mode == "launch-unconfirmed" { return }
                NSWorkspace.shared.openApplication(at: app, configuration: config) { running, error in
                    if mode == "launch-callback-timeout" { return } // Actual process launches; suppress only its callback.
                    if ["restore-slow", "restore-row-failure"].contains(mode), let running {
                        try? Data("\(running.processIdentifier)".utf8).write(to: root.appendingPathComponent("native-launch-complete"), options: .atomic)
                    }
                    completion(running, error)
                }
            }
            try await UpdateInstaller.perform(plan, limits: limits, checkpoint: { phase in
                if mode == "rollback-cleanup", phase == "rollback-exchanged" {
                    _ = chflags(plan.destination.path + ".previous", UInt32(UF_IMMUTABLE))
                }
                if mode == "death-\(phase)" {
                    try? Data("\(getpid())".utf8).write(to: root.appendingPathComponent("checkpoint"), options: .atomic)
                    raise(SIGSTOP)
                }
            }, open: opener)
            try Data("success".utf8).write(to: root.appendingPathComponent("helper-finished"), options: .atomic)
        } catch {
            try? Data(error.localizedDescription.utf8).write(to: root.appendingPathComponent("helper-finished"), options: .atomic)
            exit(1)
        }
    }
}
