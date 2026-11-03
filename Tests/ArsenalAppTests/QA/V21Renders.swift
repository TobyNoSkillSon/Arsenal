// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import CoreImage
import ArsenalCore

/// `Arsenal --render-v21 <dir>`: the v2.1 browser and table at real size (2× pixels) from the real catalog, in the three
/// appearances. ARSENAL_CATALOG / ARSENAL_SUPPORT_DIR as the app uses them (use a fresh support dir). Table membership goes
/// through the service; loaded/pending/environment states are set on the service's own rows for the picture (no GPU).
/// Translucent windows are drawn offscreen, where the window server cannot blur what is behind them: those renders put
/// a blurred desktop picture (white, and the dark Iridescence) under the window's own translucent surfaces, so they
/// check legibility, not the system material itself. Writes v21-*.png and v21-renders.txt.
@MainActor final class V21RenderDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    var exitCode: Int32 = 1
    private var log: [String] = []
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            do { try await run(); exitCode = 0 } catch { print("render-v21 FAILED: \(TableController.message(error))") }
            QA.stopApp()
        }
    }

    private func set(_ mode: RenderAppearance) async {
        Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
        try? await Task.sleep(for: .milliseconds(100))
    }

    private func run() async throws {
        let (service, error) = await ServiceFactory.make(paths: ArsenalPaths())
        if let error { throw QAFailure(error) }
        await set(.dark)

        // The browser: the two suggested picks added through it, then default, filtered, expanded, small, maximised.
        let browser = BrowserController(service: service)
        await browser.refresh()
        say("browser: \(browser.count); recommended first: " + browser.shown.prefix(4).map(\.name).joined(separator: ", "))
        for id in ["kokoro-82m", "qwen3.8-27b"] { browser.add(id) }
        try await Task.sleep(for: .milliseconds(300)); await browser.refresh()
        let window = BrowserWindowController(controller: browser)
        try await capture(window, "browser-default")
        checkHitTesting(window)

        browser.state.kind = "voices"; browser.state.toggle(input: .audio); browser.state.toggle(.licence, BrowserFacet.commercialValue)
        say("filtered (Voices, Audio in, Commercial use): " + browser.shown.map(\.name).joined(separator: ", "))
        try await capture(window, "browser-filtered")
        browser.state = BrowserState()

        browser.expanded = ["qwen3.8-27b", "kokoro-82m"]
        for id in browser.expanded {
            if let d = browser.detail(id) {
                say(
                    "detail \(id): sizes " + d.sizes.map { "\($0.dtype) \($0.size) \($0.note)" }.joined(separator: " | ") + "; runners "
                        + d.runners.map { "\($0.name) [\($0.engine)] \($0.condition)" }.joined(separator: " | "))
            }
        }
        try await capture(window, "browser-expanded")

        // The sidebar at both ends of its drag (its content follows), with a refused Remove in the count line, then the
        // collapsed toolbar showing the same error in place of the summary.
        let rootOpen = UserDefaults.standard.object(forKey: BrowserRootView.openKey), savedWidth = UserDefaults.standard.object(forKey: BrowserRootView.widthKey)
        let expandedBefore = browser.expanded; browser.expanded = []
        for width in [BrowserSidebar.minWidth, BrowserSidebar.maxWidth] {
            UserDefaults.standard.set(Double(width), forKey: BrowserRootView.widthKey)
            let sized = BrowserWindowController(controller: browser)
            browser.lastError = "Kokoro 82M: a Get is running; cancel it in the table or wait."
            try await capture(sized, "browser-sidebar-\(Int(width))", size: NSSize(width: 1100, height: 700))
            sized.close()
        }
        UserDefaults.standard.set(false, forKey: BrowserRootView.openKey)
        let collapsed = BrowserWindowController(controller: browser)
        browser.lastError = "Kokoro 82M: a Get is running; cancel it in the table or wait."
        try await capture(collapsed, "browser-collapsed-error", size: NSSize(width: 1100, height: 400))
        collapsed.close(); browser.lastError = nil
        UserDefaults.standard.set(true, forKey: BrowserRootView.openKey); browser.expanded = expandedBefore
        if let savedWidth { UserDefaults.standard.set(savedWidth, forKey: BrowserRootView.widthKey) } else { UserDefaults.standard.removeObject(forKey: BrowserRootView.widthKey) }

        browser.expanded = ["breeze-tts-2"]
        try await capture(window, "browser-small", size: NSSize(width: 920, height: 560))
        browser.expanded = ["kokoro-82m"]
        UserDefaults.standard.set(true, forKey: BrowserRootView.openKey)
        let fresh = BrowserWindowController(controller: browser)
        try await capture(fresh, "browser-max", size: NSSize(width: 1728, height: 1000))
        fresh.close()

        await set(.light)
        let light = BrowserWindowController(controller: browser)
        try await capture(light, "browser-light")
        light.close()
        await set(.translucent)
        let glass = BrowserWindowController(controller: browser)
        browser.expanded = []
        for (name, picture) in [("white", Self.whitePicture()), ("dark", Self.darkPicture())] {
            try await capture(glass, "browser-translucent-\(name)", backdrop: picture)
        }
        glass.close(); window.close()
        if let rootOpen { UserDefaults.standard.set(rootOpen, forKey: BrowserRootView.openKey) }

        // The table in each appearance: Laya loaded (blue), Qwen loaded with a load setting changed (red, Reload), FLUX
        // with missing weights (Get · 256 MB), Kokoro ready.
        for id in ["laya-english", "flux2-klein-4b"] { try await service.add(id: id) }
        let table = TableController(service: service)
        await table.refresh()
        var rows = table.rows
        func edit(_ id: String, _ change: (inout ModelRow) -> Void) { if let i = rows.firstIndex(where: { $0.id == id }) { change(&rows[i]) } }
        edit("laya-english") { r in
            r.states["fp16"] = .init(state: .loaded); r.selectedPrecision = "fp16"; r.residentRunner = r.runner
        }
        edit("qwen3.8-27b") { r in
            r.states["int4"] = .init(state: .loaded); r.selectedPrecision = "int4"; r.needsReload = true; r.residentRunner = r.runner
        }
        edit("flux2-klein-4b") { r in
            r.states["bf16"] = .init(state: .notInstalled); r.downloadBytes["bf16"] = 256_084_693
        }
        edit("kokoro-82m") { r in r.states["fp16"] = .init(state: .ready) }
        table.apply(rows: rows, kinds: table.kinds, catalog: table.catalog)
        if let i = table.settings["qwen3.8-27b"]?.firstIndex(where: { $0.applies == "load" }), let s = table.settings["qwen3.8-27b"]?[i] {
            table.settings["qwen3.8-27b"]?[i].value = s.choices.first { $0 != s.value } ?? s.value; table.settings["qwen3.8-27b"]?[i].pending = true
        }
        var tips: [String] = []
        for mode in [RenderAppearance.dark, .light, .translucent] {
            await set(mode)
            if mode == .translucent {
                for (name, picture) in [("white", Self.whitePicture()), ("dark", Self.darkPicture())] {
                    try await shot(table, "table-translucent-\(name)", backdrop: Self.blurred(picture, size: NSSize(width: 1100, height: 700)))
                }
            } else {
                try await shot(table, "table-\(mode.rawValue)")
            }
        }
        for row in table.rows {
            let p = table.presentation(row)
            tips.append(
                "[\(row.name) · Action \(p.actionTitle)\(p.pending ? " · red" : p.hot ? " · blue" : "")]\n\(ModelHelp.action(row, presentation: p, changes: table.pendingChanges(row), loadChoices: table.loadChoices(row)))\n"
            )
            tips.append("[\(row.name) · Model]\n\(ModelHelp.text(row, presentation: p))\n")
        }
        await set(.dark)
        try (tips.joined(separator: "\n") + "\n").write(to: directory.appendingPathComponent("v21-tooltips.txt"), atomically: true, encoding: .utf8)
        try log.joined(separator: "\n").write(to: directory.appendingPathComponent("v21-renders.txt"), atomically: true, encoding: .utf8)
    }

    private func say(_ line: String) { print(line); log.append(line) }

    /// Clicks in the top band must reach the header (sort, drag) and the sidebar's button, not the title bar.
    private func checkHitTesting(_ controller: BrowserWindowController) {
        guard let window = controller.window, let frame = window.contentView?.superview, let root = window.contentView as? BrowserRootView else { return }
        let band = root.band, top = frame.bounds.height
        for (what, x) in [("header", BrowserSidebar.width + 300), ("sidebar band", 120.0)] {
            let hit = frame.hitTest(NSPoint(x: x, y: top - band / 2))
            say("hit test \(what) at band centre (band \(Int(band)) pt): \(hit.map { String(describing: type(of: $0)) } ?? "nil")")
        }
    }

    private func shot(_ table: TableController, _ name: String, backdrop: NSImage? = nil) async throws {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            RenderDelegate.renderTable(table, to: directory.appendingPathComponent("v21-\(name).png"), backdrop: backdrop) { done.resume() }
        }
        say("wrote v21-\(name).png")
    }

    private func capture(_ controller: BrowserWindowController, _ name: String, size: NSSize = BrowserWindowController.defaultSize, backdrop: NSImage? = nil) async throws {
        guard let window = controller.window else { return }
        controller.applyAppearance()
        window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
        window.orderFrontRegardless()
        window.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        try await Task.sleep(for: .milliseconds(700))
        guard let rep = WindowSnapshot.capture(window) else { return }
        var output = rep
        if let backdrop, let composed = Self.compose(rep, over: Self.blurred(backdrop, size: window.frame.size)) { output = composed }
        try output.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("v21-\(name).png"))
        window.orderOut(nil)
        say("wrote v21-\(name).png")
    }

    // MARK: Desktop pictures for the translucent checks

    static func whitePicture() -> NSImage {
        NSImage(size: NSSize(width: 1600, height: 1000), flipped: false) { r in
            NSColor.white.setFill(); r.fill(); return true
        }
    }
    static func darkPicture() -> NSImage {
        NSImage(contentsOfFile: "/System/Library/Desktop Pictures/.thumbnails/Iridescence.heic")
            ?? NSImage(contentsOfFile: "/System/Library/Desktop Pictures/Solid Colors/Black.png") ?? whitePicture()
    }
    /// The system material blurs and saturates what is behind the window; this approximates it.
    static func blurred(_ image: NSImage, size: NSSize) -> NSImage {
        guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return image }
        let input = CIImage(cgImage: cg)
        let scaled = input.transformed(by: CGAffineTransform(scaleX: size.width * 2 / input.extent.width, y: size.height * 2 / input.extent.height))
        let blur = scaled.clampedToExtent().applyingGaussianBlur(sigma: 50).cropped(to: scaled.extent)
            .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 1.6])
        let rep = NSCIImageRep(ciImage: blur)
        let out = NSImage(size: size); out.addRepresentation(rep)
        return out
    }
    static func compose(_ window: NSBitmapImageRep, over backdrop: NSImage) -> NSBitmapImageRep? {
        let size = NSSize(width: window.pixelsWide, height: window.pixelsHigh)
        guard
            let out = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: window.pixelsWide, pixelsHigh: window.pixelsHigh, bitsPerSample: 8, samplesPerPixel: 4,
                hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
        else { return nil }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: out)
        let rect = NSRect(origin: .zero, size: size)
        let clip = NSBezierPath(roundedRect: rect, xRadius: 24, yRadius: 24); clip.addClip()
        backdrop.draw(in: rect)
        window.draw(in: rect, from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: false, hints: nil)
        NSGraphicsContext.restoreGraphicsState()
        out.size = window.size
        return out
    }
}

/// A window drawn offscreen at 2×: the content view (with a full-size content view the frame view's cache leaves it
/// out), the traffic lights on top, clipped to the window's rounded corners.
@MainActor enum WindowSnapshot {
    static func capture(_ window: NSWindow, corner: CGFloat = 12) -> NSBitmapImageRep? {
        guard let root = window.contentView else { return nil }
        root.layoutSubtreeIfNeeded(); window.displayIfNeeded(); root.displayIfNeeded()
        let scale: CGFloat = 2
        guard
            let out = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: Int(root.bounds.width * scale), pixelsHigh: Int(root.bounds.height * scale), bitsPerSample: 8,
                samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
        else { return nil }
        out.size = root.bounds.size
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: out)
        NSBezierPath(roundedRect: root.bounds, xRadius: corner, yRadius: corner).addClip()
        // Each top-level view on its own, in z-order: a cached display of the whole tree lets a view that clears its
        // background (a table, a scroll view) erase the fills under it, which the window server never does.
        func place(_ r: NSRect) -> NSRect { root.isFlipped ? NSRect(x: r.minX, y: root.bounds.height - r.maxY, width: r.width, height: r.height) : r }
        for v in root.subviews where !v.isHidden && v.frame.width > 0 && v.frame.height > 0 {
            if let fill = v as? FillView { fill.color.setFill(); place(v.frame).fill(using: .sourceOver); continue }
            if v is NSVisualEffectView { continue }
            if let scroll = v as? NSScrollView, let document = scroll.documentView {
                // The document's visible part and the header, each cached on its own (a scroll view's cache leaves
                // layer-backed table rows out).
                let visible = document.visibleRect
                if visible.width > 0, visible.height > 0, let rep = document.bitmapImageRepForCachingDisplay(in: visible) {
                    document.cacheDisplay(in: visible, to: rep)
                    rep.draw(
                        in: place(root.convert(scroll.contentView.bounds, from: scroll.contentView)), from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: false,
                        hints: nil)
                }
                for part in scroll.subviews where part !== scroll.contentView && !(part is NSScroller) && !part.isHidden && part.frame.height > 0 {
                    guard let rep = part.bitmapImageRepForCachingDisplay(in: part.bounds) else { continue }
                    part.cacheDisplay(in: part.bounds, to: rep)
                    rep.draw(in: place(root.convert(part.bounds, from: part)), from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: false, hints: nil)
                }
                continue
            }
            guard let rep = v.bitmapImageRepForCachingDisplay(in: v.bounds) else { continue }
            v.cacheDisplay(in: v.bounds, to: rep)
            rep.draw(in: place(v.frame), from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: false, hints: nil)
        }
        for kind in [NSWindow.ButtonType.closeButton, .miniaturizeButton, .zoomButton] {
            guard let b = window.standardWindowButton(kind), !b.isHidden, let rep = b.bitmapImageRepForCachingDisplay(in: b.bounds) else { continue }
            b.cacheDisplay(in: b.bounds, to: rep)
            rep.draw(in: b.convert(b.bounds, to: nil), from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: false, hints: nil)
        }
        NSGraphicsContext.restoreGraphicsState()
        return out
    }
}
