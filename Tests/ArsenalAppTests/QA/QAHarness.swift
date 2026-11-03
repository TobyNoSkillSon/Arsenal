// SPDX-License-Identifier: MIT
import AppKit
import Foundation
@testable import Arsenal
import Testing

// The offscreen render harness and the real (GPU) cycles, as test support: none of it ships in the app.
//
//   ARSENAL_QA="<mode> <args…>" scripts/test.sh --filter QAHarness
//
// Modes (arguments as the old app flags took them):
//   render-table <dir>                stub table states (stub-*.png)
//   render-real <dir>                 real catalog, first launch (real-*.png; no GPU)
//   render-v2 | render-v21 | render-v22 <dir>    earlier versions' surfaces from the real catalog (no GPU)
//   render-v3 <dir>                   v3 surfaces: tool windows, activity panel, menu (v3-*.png; no GPU)
//   render-dropin <dir>               the generic operation tool on the layers fixture, through the API (dropin-*.png)
//   render-sound <dir>                Make a Sound: real take through the API over the stub, previews, menu (sound-*.png)
//   render-background <dir>           Remove Background: real cut through the API over the stub, previews, menu (background-*.png)
//   render-capabilities <dir>         every capability window through the API over the fixtures, the changed windows, menu (cap-*.png)
//   render-settings <dir>             each kind's reference settings panel, dark/light, plain/busy (settings-*.png)
//   render-model-settings <dir>       the settings panel of each model in the user's table, from its own folder
//   render-withdrawn <dir>            the settings card's Remove footer for old precisions on disk (settings-withdrawn-*.png)
//   real-cycle <id> <dir>             Get → Load → run → Unload through the table (GPU: scripts/mlock gpulock, ≤ 180 s)
//   real-cycle-v2 <id> <dir> <change> | real-cycle-v21 <id> <dir> | real-cycle-v22 <id> <dir>
//   real-v3 <dir>                     chat window + activity panel on a loaded chat model, voice muted (GPU)
// Real modes need ARSENAL_SUPPORT_DIR/ARSENAL_CACHE_DIR pointing at an isolated (cloned) support dir.

@MainActor enum QAHarness {
    private static var done: CheckedContinuation<Void, Never>?
    static func finish() { done?.resume(); done = nil }

    nonisolated static var mode: [String]? {
        ProcessInfo.processInfo.environment["ARSENAL_QA"].map { $0.split(separator: " ").map(String.init) }.flatMap { $0.isEmpty ? nil : $0 }
    }

    static func delegate(_ a: [String]) -> (NSObject & NSApplicationDelegate)? {
        func dir(_ i: Int) -> URL { URL(fileURLWithPath: a[i], isDirectory: true) }
        switch (a[0], a.count) {
        case ("render-table", 2): return RenderDelegate(directory: dir(1))
        case ("render-real", 2): return RealRenderDelegate(directory: dir(1))
        case ("render-v2", 2): return V2RenderDelegate(directory: dir(1))
        case ("render-v21", 2): return V21RenderDelegate(directory: dir(1))
        case ("render-v22", 2): return V22RenderDelegate(directory: dir(1))
        case ("render-v3", 2): return V3RenderDelegate(directory: dir(1))
        case ("render-dropin", 2): return DropinRenderDelegate(directory: dir(1))
        case ("render-sound", 2): return SoundRenderDelegate(directory: dir(1))
        case ("render-background", 2): return BackgroundRenderDelegate(directory: dir(1))
        case ("render-capabilities", 2): return CapabilityRenderDelegate(directory: dir(1))
        case ("render-settings", 2): return SettingsRenderDelegate(directory: dir(1))
        case ("render-model-settings", 2): return SettingsRenderDelegate(directory: dir(1), models: true)
        case ("render-withdrawn", 2):
            let d = SettingsRenderDelegate(directory: dir(1)); d.only = SettingsRenderDelegate.withdrawnShots(); return d
        case ("real-v3", 2): return RealV3Delegate(directory: dir(1))
        case ("idle-baseline-v3", 2):
            let d = RealV3Delegate(directory: dir(1)); d.baselineOnly = true; return d
        case ("real-cycle", 3): return RealCycleDelegate(model: a[1], directory: dir(2))
        case ("real-cycle-v2", 4): return RealCycleV2Delegate(model: a[1], directory: dir(2), change: a[3])
        case ("real-cycle-v21", 3): return RealCycleV21Delegate(model: a[1], directory: dir(2))
        case ("real-cycle-v22", 3): return RealCycleV22Delegate(model: a[1], directory: dir(2))
        default: return nil
        }
    }

    /// Starts the mode's delegate as the app would and waits for its `QA.stopApp()`.
    static func run(_ a: [String]) async -> Int32 {
        guard let delegate = delegate(a) else { print("QAHarness: unknown mode \(a)"); return 2 }
        NSApplication.shared.setActivationPolicy(.accessory)
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            done = c
            delegate.applicationDidFinishLaunching?(Notification(name: NSApplication.didFinishLaunchingNotification))
        }
        return (delegate as? ExitCoding)?.exitCode ?? 0
    }
}

@MainActor @Suite(.menuState) struct QAHarnessTests {
    @Test(.enabled(if: QAHarness.mode != nil)) func qa() async {
        let code = await QAHarness.run(QAHarness.mode ?? [])
        #expect(code == 0)
    }
}
