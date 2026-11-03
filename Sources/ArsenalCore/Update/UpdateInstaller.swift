// SPDX-License-Identifier: MIT
import AppKit
import Darwin
import Foundation
import os

/// Installer runs only in Contents/Helpers/ArsenalUpdateHelper, after the controller hands off a verified release.
@MainActor
public enum UpdateInstaller {
    private static let logger = Logger(subsystem: "dev.arsenal.app", category: "UpdateInstaller")
    // Constructor-only deadlines/checkpoints let the isolated fixture exercise actual failure paths quickly.
    struct Limits: Sendable {
        var exit: TimeInterval = 120
        var lateExit: TimeInterval = 30
        var launch: TimeInterval = 60
        var restore: TimeInterval = 30 * 60
    }
    typealias Open = @MainActor (URL, NSWorkspace.OpenConfiguration, @escaping @Sendable (NSRunningApplication?, (any Error)?) -> Void) -> Void
    private static let workspaceOpen: Open = { app, configuration, completion in
        NSWorkspace.shared.openApplication(at: app, configuration: configuration, completionHandler: completion)
    }
    public static func install(_ plan: UpdateInstallPlan) async throws {
        try await perform(plan, limits: Limits())
    }
    static func perform(
        _ plan: UpdateInstallPlan, limits: Limits, checkpoint: (String) -> Void = { _ in }, open: Open = workspaceOpen
    ) async throws {
        let fm = FileManager.default
        let destination = plan.destination.standardizedFileURL
        let previous = URL(fileURLWithPath: destination.path + ".previous")
        let parent = destination.deletingLastPathComponent()
        let incoming = parent.appendingPathComponent(".Arsenal.app.incoming-\(plan.token)")
        let older = parent.appendingPathComponent(".Arsenal.app.previous-\(plan.token)")
        var swapped = false, preparedPrevious = false, movedPrevious = false, answered = false
        var launched: NSRunningApplication?
        do {
            guard destination.lastPathComponent == "Arsenal.app", destination.resolvingSymlinksInPath() == destination,
                plan.staged.directory.resolvingSymlinksInPath() == plan.staged.directory,
                plan.staged.app.path.hasPrefix(plan.staged.directory.path + "/"), UUID(uuidString: plan.token) != nil,
                !fm.fileExists(atPath: incoming.path), !fm.fileExists(atPath: older.path)
            else { throw UpdateFailure("The update install paths are invalid.") }
            reconcile(destination)
            checkpoint("after-reconcile")
            // The extracted tree is what we install. Validate its sealed contents, not its unused zip again.
            try UpdateVerification.bundle(plan.staged.app, version: plan.staged.version)
            try UpdateVerification.identity(downloaded: plan.staged.app, installed: destination)
            try fm.copyItem(at: plan.staged.app, to: incoming)
            try UpdateVerification.bundle(incoming, version: plan.staged.version)
            try UpdateVerification.identity(downloaded: incoming, installed: destination)
            // xattr -dr fails on a read-only file even without the attribute (1.0.0 shipped read-only licence copies).
            _ = try UpdateVerification.tool("/bin/chmod", ["-R", "u+w", incoming.path])
            _ = try UpdateVerification.tool("/usr/bin/xattr", ["-dr", "com.apple.quarantine", incoming.path])
            try waitForAppExit(pid: plan.parentPID, app: destination, timeout: limits.exit)
            try UpdateVerification.identity(downloaded: incoming, installed: destination)
            if fm.fileExists(atPath: previous.path) { try fm.moveItem(at: previous, to: older); movedPrevious = true }
            try fm.moveItem(at: incoming, to: previous); preparedPrevious = true
            checkpoint("before-swap")
            // At the atomic exchange, destination stays present and the old bundle becomes .previous immediately.
            try exchange(previous, destination)
            swapped = true; checkpoint("after-swap")
            try plan.persistRestore(version: plan.staged.version)
            launched = try await launch(destination, plan: plan, timeout: limits.launch, open: open)
            guard let launched else { throw UpdateFailure("The updated app did not launch.") }
            try await awaitAnswer(plan, application: launched, timeout: limits.launch)
            answered = true
            _ = await awaitRestored(plan, application: launched, timeout: limits.restore)
            try? fm.removeItem(at: older)
            cleanup(plan)
        } catch {
            let failure = error.localizedDescription
            if swapped, error is LaunchUnconfirmed {
                // LaunchServices may still finish this one request. Starting another app or rolling back its
                // on-disk bundle now would race that launch. Retain the verified app and restore record.
                try? writeFailure(
                    "Update installed, but launch was not confirmed in time. No second instance was started. Previous bundle: \(previous.path). [update:\(plan.token)]",
                    support: plan.support)
                try? fm.removeItem(at: older)
                if let app = runningApplication(at: destination), acknowledgement(plan, application: app) != nil {
                    try? fm.removeItem(at: plan.support.appendingPathComponent("update-failure.txt"))
                }
                try? fm.removeItem(at: plan.staged.directory)
            } else if swapped && !answered {
                do {
                    if let running = launched ?? runningApplication(at: destination) { try await terminate(running) }
                    // Rollback is also one atomic exchange: no missing or partially copied installed app.
                    try exchange(previous, destination); swapped = false
                    checkpoint("rollback-exchanged")
                    try? fm.removeItem(at: previous)
                    if movedPrevious { try? fm.moveItem(at: older, to: previous) }
                    try? fm.removeItem(at: UpdateAcknowledgement.url(support: plan.support, token: plan.token))
                    try? plan.persistRestore(version: plan.previousVersion)
                    try? writeFailure("Update failed: \(failure) The previous Arsenal was restored.", support: plan.support)
                    let old = try await launch(destination, plan: plan, timeout: limits.launch, open: open)
                    try await awaitAnswer(plan, application: old, timeout: limits.launch)
                    _ = await awaitRestored(plan, application: old, timeout: limits.restore)
                    cleanup(plan)
                } catch {
                    let location = swapped ? "Retained bundle: \(previous.path)." : "The previous app is installed at \(destination.path)."
                    let reason = "Update failed: \(failure) Rollback could not finish: \(error.localizedDescription) \(location)"
                    try? writeFailure(reason, support: plan.support)
                    throw UpdateFailure(reason)
                }
            } else if !swapped {
                if preparedPrevious { try? fm.removeItem(at: previous) }
                if movedPrevious { try? fm.moveItem(at: older, to: previous) }
                try? writeFailure("Update failed: \(failure) Nothing was installed; previous bundle path: \(previous.path).", support: plan.support)
                // A delayed .terminateLater reply can arrive just after the first deadline. Watch once more,
                // bounded, before giving the still-running parent its gate back via helper termination.
                if (try? waitForAppExit(pid: plan.parentPID, app: destination, timeout: limits.lateExit)) != nil {
                    try? plan.persistRestore(version: plan.previousVersion)
                    if let old = try? await launch(destination, plan: plan, timeout: limits.launch, open: open) {
                        try? await awaitAnswer(plan, application: old, timeout: limits.launch)
                        _ = await awaitRestored(plan, application: old, timeout: limits.restore)
                    }
                }
                cleanup(plan)
            } else {
                // Once the new app answers, model/runtime problems stay in that version and never force it to quit.
                cleanup(plan)
            }
            try? fm.removeItem(at: incoming)
            throw UpdateFailure(failure)
        }
    }
    /// Reconcile our UUID staging names without risking an unverified or pre-v3 backup.
    /// Recovery is best effort and never changes the installed bundle.
    static func reconcile(_ destination: URL) {
        let fm = FileManager.default, parent = destination.deletingLastPathComponent()
        func version(_ app: URL) -> UpdateVersion? {
            guard let data = try? Data(contentsOf: app.appendingPathComponent("Contents/Info.plist")),
                let info = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
                info["CFBundleIdentifier"] as? String == "dev.arsenal.app",
                let text = info["CFBundleShortVersionString"] as? String
            else { return nil }
            return UpdateVersion(text)
        }
        func removeVerified(_ app: URL) {
            guard let version = version(app),
                (try? UpdateVerification.bundle(app, version: version.description)) != nil,
                (try? UpdateVerification.identity(downloaded: app, installed: destination)) != nil
            else {
                logger.warning("Keeping unverified update backup: \(app.path, privacy: .public)"); return
            }
            if (try? fm.removeItem(at: app)) == nil { logger.warning("Could not remove update leftover: \(app.path, privacy: .public)") }
        }
        guard let currentVersion = version(destination),
            let entries = try? fm.contentsOfDirectory(at: parent, includingPropertiesForKeys: [.isSymbolicLinkKey])
        else { logger.warning("Could not inspect update leftovers."); return }
        var old: [URL] = []
        for entry in entries {
            let name = entry.lastPathComponent
            let incoming = name.hasPrefix(".Arsenal.app.incoming-")
            let prefix = incoming ? ".Arsenal.app.incoming-" : ".Arsenal.app.previous-"
            guard name.hasPrefix(prefix), UUID(uuidString: String(name.dropFirst(prefix.count))) != nil,
                let values = try? entry.resourceValues(forKeys: [.isSymbolicLinkKey]), values.isSymbolicLink != true
            else { continue }
            if incoming {
                if version(entry) != nil {
                    removeVerified(entry)
                } else if (try? fm.removeItem(at: entry)) == nil {
                    logger.warning("Could not remove incomplete update copy: \(entry.path, privacy: .public)")
                }
            } else if version(entry) != nil {
                old.append(entry)
            } else {
                logger.warning("Keeping unidentified update backup: \(entry.path, privacy: .public)")
            }
        }
        old.sort { $0.lastPathComponent < $1.lastPathComponent }
        let previous = URL(fileURLWithPath: destination.path + ".previous")
        if let recovered = old.last, version(previous) == nil || version(previous)! > currentVersion {
            // A failed verification must never erase a usable pre-v3 backup. Move uncertain bundles aside.
            if fm.fileExists(atPath: previous.path) {
                let retained = parent.appendingPathComponent(".Arsenal.app.retained-\(UUID().uuidString)")
                guard (try? fm.moveItem(at: previous, to: retained)) != nil else {
                    logger.warning("Could not retain previous update bundle: \(previous.path, privacy: .public)"); return
                }
            }
            if (try? fm.moveItem(at: recovered, to: previous)) == nil {
                logger.warning("Could not recover update backup: \(recovered.path, privacy: .public)")
            }
        }
        for entry in old where fm.fileExists(atPath: entry.path) { removeVerified(entry) }
    }

    static func exchange(_ first: URL, _ second: URL) throws {
        guard renamex_np(first.path, second.path, UInt32(RENAME_SWAP)) == 0 else {
            throw UpdateFailure("Cannot atomically exchange Arsenal with \(first.path): \(String(cString: strerror(errno))).")
        }
    }
    private static func cleanup(_ plan: UpdateInstallPlan) {
        try? FileManager.default.removeItem(at: plan.staged.directory)
        try? FileManager.default.removeItem(at: UpdateAcknowledgement.url(support: plan.support, token: plan.token))
        UpdateRestore.remove(support: plan.support, token: plan.token)
    }

    /// Kernel exit notification, not a wakeup loop. EINTR retries use the same monotonic deadline.
    static func waitForAppExit(pid: Int32, app: URL, timeout: TimeInterval = 120) throws {
        guard pid > 1, pid != getpid() else { throw UpdateFailure("The updater parent process is invalid.") }
        if kill(pid, 0) != 0, errno == ESRCH { return }
        let queue = kqueue()
        guard queue >= 0 else { throw UpdateFailure("Cannot wait for Arsenal to quit.") }
        defer { close(queue) }
        var change = kevent(ident: UInt(pid), filter: Int16(EVFILT_PROC), flags: UInt16(EV_ADD | EV_ONESHOT), fflags: UInt32(NOTE_EXIT), data: 0, udata: nil)
        if kevent(queue, &change, 1, nil, 0, nil) != 0 {
            if errno == ESRCH { return }
            throw UpdateFailure("Cannot watch Arsenal exit.")
        }
        var path = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN))
        let length = proc_pidpath(pid, &path, UInt32(path.count))
        let executable = URL(fileURLWithPath: String(decoding: path.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)).resolvingSymlinksInPath()
        if length == 0 {
            var exited = kevent(), instant = timespec(tv_sec: 0, tv_nsec: 0)
            if kevent(queue, nil, 0, &exited, 1, &instant) == 1, exited.fflags & UInt32(NOTE_EXIT) != 0 { return }
            if kill(pid, 0) != 0, errno == ESRCH { return }
        }
        guard length > 0, executable == app.appendingPathComponent("Contents/MacOS/Arsenal").resolvingSymlinksInPath() else {
            throw UpdateFailure("The updater parent is not the installed Arsenal.")
        }
        let deadline = ProcessInfo.processInfo.systemUptime + timeout
        while true {
            let remaining = max(0, deadline - ProcessInfo.processInfo.systemUptime)
            var event = kevent(), duration = timespec(tv_sec: Int(remaining), tv_nsec: Int((remaining - floor(remaining)) * 1e9))
            let result = kevent(queue, nil, 0, &event, 1, &duration)
            if result < 0, errno == EINTR, remaining > 0 { continue }
            guard result == 1, event.flags & UInt16(EV_ERROR) == 0, event.fflags & UInt32(NOTE_EXIT) != 0 else {
                throw UpdateFailure("Arsenal did not quit in time; nothing was installed.")
            }
            return
        }
    }
    /// Prefer the running instance if LaunchServices never returns its callback. Never request a second instance.
    private static func runningApplication(at app: URL) -> NSRunningApplication? {
        NSRunningApplication.runningApplications(withBundleIdentifier: "dev.arsenal.app").first {
            !$0.isTerminated && $0.bundleURL?.resolvingSymlinksInPath().path == app.resolvingSymlinksInPath().path
        }
    }
    private struct LaunchUnconfirmed: LocalizedError {
        var errorDescription: String? { "Launching Arsenal timed out without a confirmed result." }
    }
    @MainActor private final class LaunchResult {
        var continuation: CheckedContinuation<NSRunningApplication, any Error>?
        init(_ continuation: CheckedContinuation<NSRunningApplication, any Error>) { self.continuation = continuation }
        func finish(_ result: Result<NSRunningApplication, any Error>) {
            guard let continuation else { return }
            self.continuation = nil; continuation.resume(with: result)
        }
    }
    private static func launch(_ app: URL, plan: UpdateInstallPlan, timeout: TimeInterval, open: Open) async throws -> NSRunningApplication {
        guard FileManager.default.fileExists(atPath: app.appendingPathComponent("Contents/MacOS/Arsenal").path) else {
            throw UpdateFailure("Cannot relaunch Arsenal: the installed executable is missing at \(app.path).")
        }
        if let running = runningApplication(at: app), running.processIdentifier != plan.parentPID { return running }
        let config = NSWorkspace.OpenConfiguration()
        config.activates = false
        // A different installation with the same identifier (including the user's app during lab testing) must
        // never receive this launch. Force only the first instance at this path if another install exists.
        config.createsNewApplicationInstance = NSRunningApplication.runningApplications(withBundleIdentifier: "dev.arsenal.app").contains {
            !$0.isTerminated && ($0.processIdentifier == plan.parentPID || $0.bundleURL?.resolvingSymlinksInPath().path != app.resolvingSymlinksInPath().path)
        }
        config.arguments = ["--arsenal-update", plan.token]
        var environment = ProcessInfo.processInfo.environment
        environment["ARSENAL_SUPPORT_DIR"] = plan.support.path
        config.environment = environment
        return try await withCheckedThrowingContinuation { continuation in
            let once = LaunchResult(continuation)
            let timer = Task { @MainActor in
                try? await Task.sleep(for: .seconds(timeout))
                if !Task.isCancelled {
                    if let running = runningApplication(at: app) { once.finish(.success(running)) } else { once.finish(.failure(LaunchUnconfirmed())) }
                }
            }
            open(app, config) { application, error in
                Task { @MainActor in
                    timer.cancel()
                    if let application {
                        guard application.bundleURL?.resolvingSymlinksInPath().path == app.resolvingSymlinksInPath().path else {
                            once.finish(
                                .failure(
                                    UpdateFailure("LaunchServices returned a different Arsenal installation: \(application.bundleURL?.path ?? "unknown") instead of \(app.path).")))
                            return
                        }
                        once.finish(.success(application))
                    } else {
                        once.finish(.failure(error ?? UpdateFailure("Arsenal did not launch.")))
                    }
                }
            }
        }
    }
    private static func acknowledgement(_ plan: UpdateInstallPlan, application: NSRunningApplication) -> UpdateAcknowledgement? {
        guard let data = try? Data(contentsOf: UpdateAcknowledgement.url(support: plan.support, token: plan.token)),
            let ack = try? JSONDecoder().decode(UpdateAcknowledgement.self, from: data),
            ack.token == plan.token, ack.pid == application.processIdentifier
        else { return nil }
        return ack
    }
    private static func awaitAnswer(_ plan: UpdateInstallPlan, application: NSRunningApplication, timeout: TimeInterval) async throws {
        let deadline = ProcessInfo.processInfo.systemUptime + timeout
        while ProcessInfo.processInfo.systemUptime < deadline {
            if application.isTerminated { throw UpdateFailure("The updated app exited before answering the updater.") }
            if acknowledgement(plan, application: application) != nil { return }
            try await Task.sleep(for: .seconds(1))
        }
        throw UpdateFailure("The updated app did not answer within the launch deadline.")
    }
    private static func awaitRestored(_ plan: UpdateInstallPlan, application: NSRunningApplication, timeout: TimeInterval) async -> String? {
        let deadline = ProcessInfo.processInfo.systemUptime + timeout
        while ProcessInfo.processInfo.systemUptime < deadline {
            if application.isTerminated { return "Update installed, but Arsenal exited while restoring models. The previous bundle is retained." }
            if let ack = acknowledgement(plan, application: application), ack.phase == .restored {
                return ack.reason.map { "Update installed, but loaded models could not be restored: \($0)" }
            }
            try? await Task.sleep(for: .seconds(1))
        }
        return "Update installed, but model restoration exceeded its deadline (normally thirty minutes). The previous bundle is retained."
    }
    private static func terminate(_ app: NSRunningApplication) async throws {
        _ = app.terminate()
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while !app.isTerminated && ProcessInfo.processInfo.systemUptime < deadline { try? await Task.sleep(for: .seconds(1)) }
        if !app.isTerminated { _ = app.forceTerminate() }
        let forced = ProcessInfo.processInfo.systemUptime + 5
        while !app.isTerminated && ProcessInfo.processInfo.systemUptime < forced { try? await Task.sleep(for: .seconds(1)) }
        guard app.isTerminated else { throw UpdateFailure("The updated app could not be stopped for rollback.") }
    }
    private static func writeFailure(_ reason: String, support: URL) throws {
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        try Data(reason.utf8).write(to: support.appendingPathComponent("update-failure.txt"), options: .atomic)
    }
}
