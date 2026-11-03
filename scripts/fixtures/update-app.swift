// SPDX-License-Identifier: MIT
// Compiled only by test-updates.py; never included in the shipped app.
import AppKit
import Foundation

@MainActor final class FixtureTable { var admitting = true }
@MainActor final class FixtureDelegate: NSObject, NSApplicationDelegate {
    var quitting = false
    let table = FixtureTable()
    var replyToTerminate: @MainActor (Bool) -> Void = { NSApp.reply(toApplicationShouldTerminate: $0) }
    func shutDown() async -> Bool {
        let env = ProcessInfo.processInfo.environment
        let delay: Double = env["ARSENAL_UPDATE_TEST_MODE"] == "late-exit" ? 7 : 0.25
        try? await Task.sleep(for: .seconds(delay))
        return true
    }
    func applicationDidFinishLaunching(_ notification: Notification) {
        if let root = ProcessInfo.processInfo.environment["ARSENAL_UPDATE_TEST_ROOT"],
            !["launch-failure", "launch-timeout", "rollback-cleanup"].contains(ProcessInfo.processInfo.environment["ARSENAL_UPDATE_TEST_MODE"] ?? "") {
            try? UpdateController.acknowledgeRelaunch(support: URL(fileURLWithPath: root).appendingPathComponent("support"))
        }
        Task { await UpdateFixtureApp.run() }
    }
    // applicationShouldTerminate is inserted verbatim from the production AppDelegate by the test builder.
}

@main struct UpdateFixtureApp {
    @MainActor static var retainedController: UpdateController?
    @MainActor static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let delegate = FixtureDelegate()
        app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }
    @MainActor static func run() async {
        let environment = ProcessInfo.processInfo.environment
        guard let rootPath = environment["ARSENAL_UPDATE_TEST_ROOT"] else { exit(64) }
        let root = URL(fileURLWithPath: rootPath), support = root.appendingPathComponent("support")
        let mode = environment["ARSENAL_UPDATE_TEST_MODE"] ?? "good"
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "missing"
        func write(_ name: String, _ text: String) { try? Data(text.utf8).write(to: root.appendingPathComponent(name), options: .atomic) }
        let hooks = UpdateServiceHooks(
            acquireIdle: {
                try Data("ready".utf8).write(to: root.appendingPathComponent("waiting"), options: .atomic)
                while !FileManager.default.fileExists(atPath: root.appendingPathComponent("idle").path) { try await Task.sleep(for: .milliseconds(100)) }
                return ["fixture-chat", "fixture-voice"]
            }, releaseIdle: {
                try? Data("released".utf8).write(to: root.appendingPathComponent("released"), options: .atomic)
            }, restore: { models in
                if mode == "restore-failure", version == "3.0.0" { throw UpdateFailure("Fixture model could not load.") }
                if mode == "restore-timeout", version == "3.0.0" { try await Task.sleep(for: .seconds(7)) }
                if ["restore-slow", "restore-row-failure"].contains(mode), version == "3.0.0" {
                    try Data("started".utf8).write(to: root.appendingPathComponent("restore-started"), options: .atomic)
                    if mode == "restore-slow" {
                        let deadline = ProcessInfo.processInfo.systemUptime + 20
                        while !FileManager.default.fileExists(atPath: root.appendingPathComponent("finish-restore").path) {
                            guard ProcessInfo.processInfo.systemUptime < deadline else { throw UpdateFailure("The fixture restore gate timed out.") }
                            try await Task.sleep(for: .milliseconds(100))
                        }
                    } else {
                        // Match the real non-throwing service adapter: errors belong to individual rows.
                        let failures = try JSONSerialization.data(withJSONObject: Dictionary(uniqueKeysWithValues: models.map { ($0, "Fixture model refused restoration.") }))
                        try failures.write(to: root.appendingPathComponent("row-failures"), options: .atomic)
                    }
                }
                let restoredModels = mode == "restore-row-failure" && version == "3.0.0" ? [] : models
                let data = try JSONSerialization.data(withJSONObject: ["version": version, "models": restoredModels, "pid": getpid()])
                try data.write(to: root.appendingPathComponent("restored"), options: .atomic)
            })
        let defaults = UserDefaults(suiteName: "dev.arsenal.update-test.\(root.deletingLastPathComponent().lastPathComponent).\(root.lastPathComponent)")!
        let source = environment["ARSENAL_UPDATE_TEST_API"].flatMap(URL.init(string:)).flatMap { try? UpdateSource(testServer: $0) }
        let controller = UpdateController(
            installedApp: Bundle.main.bundleURL, support: support, hooks: hooks, repository: nil,
            currentVersion: version, defaults: defaults, testSource: ["no-repo", "idle-baseline"].contains(mode) ? nil : source,
            quit: {
                write("handoff", "deferred")
                if mode != "exit-timeout" {
                    RunLoop.main.perform(inModes: [.default]) { MainActor.assumeIsolated { NSApp.terminate(nil) } }
                }
            })
        retainedController = controller
        if ProcessInfo.processInfo.arguments.contains("--arsenal-update") {
            write("launched", "\(getpid())")
            let count = root.appendingPathComponent("launch-pids")
            let prior = (try? String(contentsOf: count, encoding: .utf8)) ?? ""
            try? Data((prior + "\(getpid())\n").utf8).write(to: count, options: .atomic)
            if version == "3.0.0", mode == "slow-startup" { try? await Task.sleep(for: .seconds(7)) }
            if version == "3.0.0", mode == "restore-missing" { try? FileManager.default.removeItem(at: support.appendingPathComponent("update-restore.json")) }
            if version == "3.0.0", mode == "restore-mismatch" {
                try? JSONEncoder().encode(UpdateRestore(token: UUID().uuidString, version: version, models: ["wrong-record"])).write(
                    to: support.appendingPathComponent("update-restore.json"), options: .atomic)
            }
            if version == "3.0.0", ["launch-failure", "rollback-cleanup"].contains(mode) { exit(9) }
            if version == "3.0.0", mode == "launch-timeout" { return }
            await controller.restoreAfterRelaunch()
            write("restore-state", String(describing: controller.state))
            return
        }
        if mode == "idle-scheduler" { defaults.set(Date(), forKey: "update.lastCheck.v3") }
        controller.start()
        let stream = controller.events()
        Task { @MainActor in
            for await state in stream {
                write("state", String(describing: state))
                if case .failed(let reason) = state { write("failure", reason) }
            }
        }
        if ["idle-scheduler", "idle-baseline"].contains(mode) {
            write("armed", "\(getpid()) \(controller.schedulerArmed)")
            return
        }
        await controller.checkForUpdates(background: true)
        await controller.checkForUpdates(background: true)
        if mode == "no-repo" { write("no-repo", "\(controller.enabled) \(controller.schedulerArmed)"); exit(0) }
        if case .available = controller.state {
            write("available", "available")
            if ["check-race", "check-failure", "check-manual-failure"].contains(mode) {
                write("check-start", "ready")
                let now = mode == "check-manual-failure" ? Date() : Date().addingTimeInterval(UpdateController.checkInterval + 1)
                let check = Task { await controller.checkForUpdates(now: now, background: mode == "check-failure") }
                try? await Task.sleep(for: .milliseconds(300))
                if mode == "check-race" {
                    await controller.installAvailableUpdate()
                    await controller.installAvailableUpdate()
                    write("race-state", String(describing: controller.state))
                }
                await check.value
                write("check-state", String(describing: controller.state))
                write("check-error", controller.lastCheckFailure ?? "none")
                return
            }
            await controller.installAvailableUpdate()
            // A returning quit closure is a handoff, not a failed/cancelled update.
            write("returned", String(describing: controller.state))
        }
    }
}
