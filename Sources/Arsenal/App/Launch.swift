// SPDX-License-Identifier: MIT
import AppKit
import Darwin
import ArsenalCore

/// One app per support directory. The retained descriptor releases its lock on exit or a crash;
/// the lock file stays in place so every contender always locks the same inode.
@MainActor enum SingleInstance {
    private static var lease: InstanceLease?
    static func claim(support: URL = ArsenalPaths().support) -> Bool {
        do {
            guard let acquired = try InstanceLease.acquire(support: support) else { return false }
            lease = acquired; return true
        } catch {
            FileHandle.standardError.write(Data("Arsenal could not lock its support directory: \(error.localizedDescription)\n".utf8))
            return false
        }
    }
}
private final class InstanceLease {
    private let descriptor: Int32
    private init(_ descriptor: Int32) { self.descriptor = descriptor }
    deinit { close(descriptor) }
    static func acquire(support: URL) throws -> InstanceLease? {
        let resolved = support.standardizedFileURL.resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: resolved, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let file = resolved.appendingPathComponent(".instance.lock")
        // No truncation or PID write: a second launch only tries the lock and never changes status/config.
        // CLOEXEC prevents a worker from retaining the app's lease after the app has crashed.
        let descriptor = Darwin.open(file.path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let failure = errno; close(descriptor)
            if failure == EWOULDBLOCK || failure == EAGAIN { return nil }
            throw POSIXError(POSIXErrorCode(rawValue: failure) ?? .EIO)
        }
        return InstanceLease(descriptor)
    }
}

/// Opens the service once the app has launched (it is async: the catalog is verified first), then hands over to
/// AppDelegate. An app whose catalog cannot be opened still starts: empty table, the reason in its footer.
@MainActor final class LaunchDelegate: NSObject, NSApplicationDelegate {
    private var app: AppDelegate?
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        TerminationSignal.install { AppModal.quit() }
        // Quit works from the icon's first frame, including when catalog verification is slow or fails.
        let status = AppDelegate.makeStatusItem()
        status.menu = Self.startupMenu()
        // Before the service opens: its first launch writes into the support folder, which would hide a fresh install. The
        // default is marked owed here, so a launch cut short before its login decision still gets it on the next launch.
        let paths = ArsenalPaths()
        if AppDelegate.isFreshInstall(support: paths.support), AppDelegate.launchDefaultApplies() {
            AppDelegate.markLoginDefaultOwed(support: paths.support)
        }
        Task {
            let (service, error) = await Self.open(paths: paths)
            let app = AppDelegate(service: service, paths: paths)
            app.table.lastError = error
            app.lastOutput = LastOutput(textFile: paths.support.appendingPathComponent("state/last-output.txt"))
            app.status = status
            self.app = app
            app.applicationDidFinishLaunching(notification)
        }
    }
    /// After an update's relaunch the updater's helper waits for this launch's receipt; it is written before the
    /// catalog is verified, so a slow verification is never taken for a failed update (and rolled back).
    static func open(
        paths: ArsenalPaths, arguments: [String] = CommandLine.arguments,
        acknowledge: (URL, [String]) throws -> Void = { try UpdateController.acknowledgeRelaunch(support: $0, arguments: $1) },
        make: (ArsenalPaths) async -> (any ModelsService, String?) = { await ServiceFactory.make(paths: $0) }
    ) async -> (any ModelsService, String?) {
        try? acknowledge(paths.support, arguments)
        return await make(paths)
    }

    static func startupMenu() -> NSMenu {
        let menu = NSMenu(); menu.autoenablesItems = false
        let verifying = NSMenuItem(title: "Verifying model catalog\u{2026}", action: nil, keyEquivalent: "")
        verifying.isEnabled = false
        menu.addItem(verifying)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Arsenal", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        quit.target = NSApplication.shared
        quit.image = NSImage(systemSymbolName: "power", accessibilityDescription: nil)
        quit.showMenuImage()
        menu.addItem(quit)
        return menu
    }
    /// Before the service is open nothing is supervised yet: quit at once.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        app?.applicationShouldTerminate(sender) ?? .terminateNow
    }
    func applicationWillTerminate(_ notification: Notification) { app?.applicationWillTerminate(notification) }
}

/// The service the app runs on: the stub with `ARSENAL_STUB=1`, else core's `RealModelsService` over the signed
/// catalog in the app bundle's Resources. `ARSENAL_CATALOG` names a local catalog repository (Resources/catalog/, Resources/kinds/,
/// Resources/catalog-public-key) to use instead; the bundle stays the fallback, so a missing or dirty repository opens
/// the verified bundle with a footer line rather than an empty table. `ARSENAL_HELPERS` overrides the worker binaries'
/// folder (dev builds).
enum ServiceFactory {
    static func make(paths: ArsenalPaths, environment: [String: String] = ProcessInfo.processInfo.environment, bundle: URL? = Bundle.main.resourceURL) async -> (
        any ModelsService, String?
    ) {
        if environment["ARSENAL_STUB"] == "1" { return (FixtureService(rows: Fixtures.rows(.firstLaunch)), nil) }
        let override = environment["ARSENAL_CATALOG"].map { URL(fileURLWithPath: $0) }
        let resources = resources(bundle: bundle, override: override)
        let helpers = environment["ARSENAL_HELPERS"].map { URL(fileURLWithPath: $0) }
        do {
            try paths.create()
            let service = try await RealModelsService.open(
                repository: override ?? resources, resources: resources, paths: paths,
                artifacts: BundledWorkerArtifact(helpers: helpers), explicitCatalogRepository: override)
            // A gated model's Get opens its Hugging Face page (and the token page, without a token) in the default browser.
            await service.setGateHandling(open: HuggingFacePages.open)
            // A local folder's pinned weights files are checked against Hugging Face before approval and Get.
            let hub = HuggingFaceClient()
            await service.setPinnedFilesLister { repo, revision, type in await hub.listing(repo: repo, revision: revision, repoType: type) }
            return (service, nil)
        } catch {
            // No invented rows: an empty table whose footer says why.
            return (FixtureService(rows: []), "The model catalog could not be opened: \(TableController.message(error))")
        }
    }
    /// The signed catalog the app falls back to: the bundle's Resources whenever they hold one, also under an override;
    /// only a bare dev binary (no bundle catalog) reads everything from the override repository, as before.
    static func resources(bundle: URL?, override: URL?) -> URL {
        if let bundle, FileManager.default.fileExists(atPath: bundle.appendingPathComponent("catalog/catalog.json").path) { return bundle }
        let candidate = override ?? bundle ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let resources = candidate.appendingPathComponent("Resources")
        return FileManager.default.fileExists(atPath: resources.appendingPathComponent("catalog/catalog.json").path) ? resources : candidate
    }
}
