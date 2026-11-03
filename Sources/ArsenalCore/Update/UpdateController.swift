// SPDX-License-Identifier: MIT
import Foundation
import Observation
import os

/// Service owner must atomically drain ongoing work, block new requests, and return the loaded model IDs.
/// releaseIdle reopens admission if a handoff fails. restore calls service.restoreAfterUpdate(ids:)
/// so automatic residency is preserved; it must not use the public, manual Load operation.
public struct UpdateServiceHooks: Sendable {
    public let acquireIdle: @Sendable () async throws -> [String]
    public let releaseIdle: @Sendable () async -> Void
    public let restore: @Sendable ([String]) async throws -> Void
    public init(
        acquireIdle: @escaping @Sendable () async throws -> [String], releaseIdle: @escaping @Sendable () async -> Void,
        restore: @escaping @Sendable ([String]) async throws -> Void
    ) {
        self.acquireIdle = acquireIdle; self.releaseIdle = releaseIdle; self.restore = restore
    }
}

/// App binds to state (Observation) or events(). The controller owns no timer, and does nothing when the repo is unset.
@MainActor @Observable
public final class UpdateController {
    public private(set) var state: UpdateState = .none
    /// A failed background check preserves the current offer; this reason is for diagnostics.
    public private(set) var lastCheckFailure: String?
    public var enabled: Bool { source != nil && currentVersion != nil }
    public var schedulerArmed: Bool { scheduler != nil }
    public static let checkInterval: TimeInterval = 3 * 24 * 60 * 60
    @ObservationIgnored private let source: UpdateSource?
    @ObservationIgnored private let currentVersion: UpdateVersion?
    @ObservationIgnored private let installedApp: URL
    @ObservationIgnored private let support: URL
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let hooks: UpdateServiceHooks
    @ObservationIgnored private let quit: @MainActor () -> Void
    @ObservationIgnored private var scheduler: NSBackgroundActivityScheduler?
    @ObservationIgnored private var checking = false
    @ObservationIgnored private var helperProcess: Process?
    @ObservationIgnored private var helperToken: String?
    @ObservationIgnored private var release: UpdateRelease?
    @ObservationIgnored private var streams: [UUID: AsyncStream<UpdateState>.Continuation] = [:]
    private static let lastCheckKey = "update.lastCheck.v3"
    private static let logger = Logger(subsystem: "dev.arsenal.app", category: "Update")
    private static let offerKey = "update.offer.v3"
    private static func failureReason(_ reason: String) -> String {
        String(reason.components(separatedBy: " [update:").first!.prefix(4000))
    }

    public init(
        installedApp: URL = Bundle.main.bundleURL, support: URL, hooks: UpdateServiceHooks,
        repository: String? = Bundle.main.object(forInfoDictionaryKey: "ArsenalUpdateRepo") as? String,
        currentVersion: String? = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
        defaults: UserDefaults = .standard, testSource: UpdateSource? = nil, quit: @escaping @MainActor () -> Void
    ) {
        self.installedApp = installedApp; self.support = support; self.hooks = hooks; self.defaults = defaults; self.quit = quit
        self.currentVersion = currentVersion.flatMap(UpdateVersion.init)
        if let testSource {
            source = testSource
        } else if let repository, !repository.isEmpty {
            do { source = try UpdateSource(repository: repository) } catch { source = nil; state = .failed(reason: error.localizedDescription) }
        } else {
            source = nil
        }
    }

    public func events() -> AsyncStream<UpdateState> {
        let id = UUID(), pair = AsyncStream<UpdateState>.makeStream(bufferingPolicy: .bufferingNewest(1))
        streams[id] = pair.continuation; pair.continuation.yield(state)
        pair.continuation.onTermination = { [weak self] _ in Task { @MainActor in self?.streams.removeValue(forKey: id) } }
        return pair.stream
    }
    private func set(_ state: UpdateState) { self.state = state; for stream in streams.values { stream.yield(state) } }

    public func start() {
        let failureURL = support.appendingPathComponent("update-failure.txt")
        let previousFailure = (try? String(contentsOf: failureURL, encoding: .utf8)).map(Self.failureReason)
        if previousFailure != nil { try? FileManager.default.removeItem(at: failureURL) }
        guard enabled, scheduler == nil else {
            if let previousFailure { set(.failed(reason: previousFailure)) }; return
        }
        if let data = defaults.data(forKey: Self.offerKey), let saved = try? JSONDecoder().decode(UpdateRelease.self, from: data),
            let version = UpdateVersion(saved.version), let currentVersion, version > currentVersion
        {
            release = saved; set(.available(version: saved.version, notes: saved.notes, size: saved.size))
        }
        if let previousFailure { set(.failed(reason: previousFailure)) }
        if release == nil { defaults.removeObject(forKey: Self.offerKey) }
        armScheduler()
    }
    public func stop() { scheduler?.invalidate(); scheduler = nil }
    private func armScheduler() {
        guard enabled else { return }
        let activity = NSBackgroundActivityScheduler(identifier: "dev.arsenal.app.release-check")
        activity.repeats = true; activity.qualityOfService = .background
        configureScheduler(activity); scheduler = activity
        let schedulerID = ObjectIdentifier(activity)
        activity.schedule { [weak self] completion in
            Task { @MainActor in
                guard let self, let activity = self.scheduler, ObjectIdentifier(activity) == schedulerID else { completion(.finished); return }
                await self.checkForUpdates(background: true)
                // Keep one native scheduler. Adjust its next interval before completion, as Foundation specifies.
                if self.scheduler === activity { self.configureScheduler(activity) }
                completion(.finished)
            }
        }
    }
    private func configureScheduler(_ activity: NSBackgroundActivityScheduler) {
        let last = defaults.object(forKey: Self.lastCheckKey) as? Date
        let remaining = last.map { Self.checkInterval - Date().timeIntervalSince($0) } ?? 0
        if remaining > 0 {
            // Tolerance is symmetric; offset the interval so the earliest OS window is the three-day due time.
            activity.interval = remaining + 3 * 60 * 60; activity.tolerance = 3 * 60 * 60
        } else {
            activity.interval = 600; activity.tolerance = 300
        }
    }
    nonisolated public static func checkIsDue(last: Date?, now: Date) -> Bool {
        guard let last else { return true }; return now.timeIntervalSince(last) >= 3 * 24 * 60 * 60
    }

    /// OS checks observe the persisted three-day limit. An explicit manual retry checks immediately.
    public func checkForUpdates(now: Date = Date(), background: Bool = false) async {
        guard let source, let currentVersion, !checking, !state.busy,
            !background || Self.checkIsDue(last: defaults.object(forKey: Self.lastCheckKey) as? Date, now: now)
        else { return }
        checking = true; defaults.set(now, forKey: Self.lastCheckKey)
        defer { checking = false }
        let network = UpdateNetwork(source: source)
        defer { network.close() }
        do {
            let found = try await network.latest()
            guard !state.busy else { return }
            lastCheckFailure = nil
            if let found, let version = UpdateVersion(found.version), version > currentVersion {
                release = found; defaults.set(try JSONEncoder().encode(found), forKey: Self.offerKey)
                set(.available(version: found.version, notes: found.notes, size: found.size))
            } else {
                release = nil; defaults.removeObject(forKey: Self.offerKey); set(.none)
            }
        } catch {
            lastCheckFailure = "Could not check for updates: \(error.localizedDescription)"
            Self.logger.warning("Update check failed: \(error.localizedDescription, privacy: .public)")
            if !background { set(.failed(reason: lastCheckFailure ?? error.localizedDescription)) }
        }
    }

    public func installAvailableUpdate() async {
        guard enabled, !checking, !state.busy, let release, let source, let currentVersion else { return }
        set(.downloading(progress: 0))
        let network = UpdateNetwork(source: source)
        defer { network.close() }
        var staged: StagedArsenalUpdate?, acquired = false
        var token: String?
        do {
            let prepared = try await UpdateVerification.prepare(release, network: network, installed: installedApp) { [weak self] progress in
                Task { @MainActor in
                    guard let self, case .downloading = self.state else { return }
                    self.set(.downloading(progress: progress))
                }
            }
            staged = prepared; set(.ready)
            let models = try await hooks.acquireIdle(); acquired = true
            try Task.checkCancellation()
            let plan = UpdateInstallPlan(
                staged: prepared, destination: installedApp, previousVersion: currentVersion.description, parentPID: getpid(), support: support, loadedModels: models)
            token = plan.token
            try plan.persistRestore(version: plan.staged.version)
            let planURL = prepared.directory.appendingPathComponent("plan.json")
            try JSONEncoder().encode(plan).write(to: planURL, options: .atomic)
            let helper = installedApp.appendingPathComponent("Contents/Helpers/ArsenalUpdateHelper")
            guard FileManager.default.isExecutableFile(atPath: helper.path) else { throw UpdateFailure("The updater helper is missing from Arsenal.app.") }
            let process = Process(); process.executableURL = helper; process.arguments = ["--plan", planURL.path]
            process.standardInput = FileHandle.nullDevice; process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
            process.terminationHandler = { [weak self] _ in
                Task { @MainActor in await self?.helperExited() }
            }
            try process.run(); helperProcess = process; helperToken = plan.token; set(.installing)
            // AppKit may return or spin its .terminateLater run loop. The helper owns the bounded exit wait.
            // The app owner MUST defer NSApp.terminate using RunLoop.main.perform, outside the main queue drain.
            quit()
        } catch {
            if acquired { await hooks.releaseIdle() }
            if let staged { try? FileManager.default.removeItem(at: staged.directory) }
            if let token { UpdateRestore.remove(support: support, token: token) }
            set(.failed(reason: error.localizedDescription))
        }
    }

    private func helperExited() async {
        guard helperProcess != nil else { return }
        helperProcess = nil
        let token = helperToken; helperToken = nil
        // This runs only if the parent is still alive (for example, quit was cancelled).
        await hooks.releaseIdle()
        if let token { UpdateRestore.remove(support: support, token: token) }
        let file = support.appendingPathComponent("update-failure.txt")
        let reason = (try? String(contentsOf: file, encoding: .utf8)) ?? "The update helper stopped before Arsenal quit; nothing was installed."
        try? FileManager.default.removeItem(at: file)
        set(.failed(reason: Self.failureReason(reason)))
    }

    /// Call as soon as the app launches, before waiting for settings/catalog/service initialization.
    /// This receipt does not depend on a restore file: missing model state cannot turn a working launch into rollback.
    public static func acknowledgeRelaunch(support: URL, arguments: [String] = ProcessInfo.processInfo.arguments) throws {
        guard let index = arguments.firstIndex(of: "--arsenal-update"), arguments.indices.contains(index + 1) else { return }
        let token = arguments[index + 1]
        guard UUID(uuidString: token) != nil else { throw UpdateFailure("The update launch token is invalid.") }
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try JSONEncoder().encode(UpdateAcknowledgement(token: token, pid: getpid(), phase: .launched)).write(
            to: UpdateAcknowledgement.url(support: support, token: token), options: .atomic)
        let failure = support.appendingPathComponent("update-failure.txt")
        if let reason = try? String(contentsOf: failure, encoding: .utf8), reason.contains("[update:\(token)]") {
            try? FileManager.default.removeItem(at: failure)
        }
    }

    /// Restore after service startup. The separate early launch acknowledgement comes first.

    public func restoreAfterRelaunch(arguments: [String] = ProcessInfo.processInfo.arguments) async {
        guard let index = arguments.firstIndex(of: "--arsenal-update"), arguments.indices.contains(index + 1) else { return }
        let token = arguments[index + 1]
        guard UUID(uuidString: token) != nil else { set(.failed(reason: "The update launch token is invalid.")); return }
        let url = support.appendingPathComponent(UpdateRestore.fileName)
        func acknowledge(_ phase: UpdateAcknowledgement.Phase, reason: String? = nil) throws {
            // The helper removes this receipt after its restore deadline. A late model completion must not recreate it.
            if phase == .restored, !FileManager.default.fileExists(atPath: UpdateAcknowledgement.url(support: support, token: token).path) { return }
            try JSONEncoder().encode(UpdateAcknowledgement(token: token, pid: getpid(), phase: phase, reason: reason)).write(
                to: UpdateAcknowledgement.url(support: support, token: token), options: .atomic)
        }
        do {
            if !FileManager.default.fileExists(atPath: UpdateAcknowledgement.url(support: support, token: token).path) {
                try Self.acknowledgeRelaunch(support: support, arguments: arguments)
            }
            let restore = try JSONDecoder().decode(UpdateRestore.self, from: Data(contentsOf: url))
            guard restore.token == token, restore.version == currentVersion?.description else { throw UpdateFailure("The update restore record does not match this launch.") }
            defaults.removeObject(forKey: Self.offerKey); release = nil
            try await hooks.restore(restore.models)
            try acknowledge(.restored)
            UpdateRestore.remove(support: support, token: token)
        } catch {
            try? acknowledge(.restored, reason: error.localizedDescription)
            UpdateRestore.remove(support: support, token: token)
            set(.failed(reason: "Could not restore loaded models: \(error.localizedDescription)"))
        }
    }
}

extension UpdateState {
    public var busy: Bool {
        switch self {
        case .downloading, .ready, .installing: return true
        default: return false
        }
    }
}
