// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore

// The menu's view of the app updater (C8). The updater itself is core's UpdateController; this is the state the
// orange item draws, and the two things it can ask for.

enum AppUpdateState: Equatable {
    case none
    case available(version: String, notes: String, size: Int64)
    case downloading(progress: Double)
    case ready
    case installing
    case failed(reason: String)
}

@MainActor protocol AppUpdating: AnyObject {
    var updateState: AppUpdateState { get }
    /// The orange item: download, verify, wait for idle, install, relaunch.
    func installUpdate()
    /// After a failure: check again.
    func checkAgain()
    var onChange: () -> Void { get set }
}

/// Core's UpdateController behind the menu item. Present only when this build names a release repository
/// (`ArsenalUpdateRepo`); otherwise there is no item, no scheduler and no network.
@MainActor final class CoreUpdater: AppUpdating {
    let controller: UpdateController
    var onChange: () -> Void = {}
    private var watching: Task<Void, Never>?

    /// nil when the build has no update repository.
    static func make(service: any ModelsService, support: URL) -> CoreUpdater? {
        let hooks = UpdateServiceHooks(
            acquireIdle: { try await service.beginUpdate() }, releaseIdle: { await service.cancelUpdate() },
            // Each model comes back with its dtype, Fast/Exact, residency and Keep Hot; failures show on its row.
            restore: { ids in await service.restoreAfterUpdate(ids: ids) })
        // Quit through the normal path (applicationShouldTerminate → .terminateLater → reply) once this call returns:
        // terminating inside the main queue's drain would wait on that same queue forever.
        let controller = UpdateController(
            support: support, hooks: hooks, quit: { RunLoop.main.perform(inModes: [.default, .modalPanel]) { MainActor.assumeIsolated { AppModal.quit() } } })
        return controller.enabled ? CoreUpdater(controller) : nil
    }

    init(_ controller: UpdateController) {
        self.controller = controller
        watching = Task { [weak self] in
            for await _ in controller.events() { self?.onChange() }
        }
        controller.start()
    }

    var updateState: AppUpdateState {
        switch controller.state {
        case .none: return .none
        case .available(let version, let notes, let size): return .available(version: version, notes: notes, size: size)
        case .downloading(let progress): return .downloading(progress: progress)
        case .ready: return .ready
        case .installing: return .installing
        case .failed(let reason): return .failed(reason: reason)
        }
    }
    func installUpdate() { Task { await controller.installAvailableUpdate() } }
    /// A failed download retries with the retained release; a failed check checks again (within the three-day limit).
    func checkAgain() {
        Task {
            await controller.installAvailableUpdate()
            if case .failed = controller.state { await controller.checkForUpdates() }
        }
    }
    /// After a relaunch by the updater: reload what was loaded (no-op on a normal launch).
    func restoreAfterRelaunch() async { await controller.restoreAfterRelaunch() }
    func stop() { controller.stop(); watching?.cancel() }
}
