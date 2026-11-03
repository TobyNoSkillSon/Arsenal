// SPDX-License-Identifier: MIT
import AppKit

/// Only these fixed identifiers may reach the log; visible titles can contain private saved-voice names.
enum MenuDiagnosticName: String, Sendable {
    case main = "Main", models = "Models", activity = "Activity", voice = "Voice"
    case keepHot = "Keep Hot", memory = "Memory", submenu = "Submenu", settings = "Settings"
    var itemIdentifier: NSUserInterfaceItemIdentifier { NSUserInterfaceItemIdentifier(rawValue) }
}

/// Bounded open-time evidence. No polling, UI, screenshots or model/request contents.
@MainActor final class MenuDiagnostics: NSObject, NSMenuDelegate {
    let url: URL
    nonisolated static let maximumBytes = 64 * 1024
    private var terminating = false
    nonisolated static let terminationBudget: DispatchTimeInterval = .milliseconds(50)
    private var pending: [ObjectIdentifier: (menu: NSMenu, name: MenuDiagnosticName)] = [:]
    private lazy var tracking = MenuTrackingRecorder(diagnostics: self)
    private let writer: any MenuGeometryWriting
    private(set) var lastError: String?
    init(support: URL, writer: (any MenuGeometryWriting)? = nil, onDiskAccess: (@Sendable () -> Void)? = nil) {
        url = support.appendingPathComponent("logs/menu-geometry.jsonl")
        self.writer = writer ?? MenuGeometryWriter(url: url, onDiskAccess: onDiskAccess)
        super.init()
    }

    func observeSubmenus(in menu: NSMenu) {
        for item in menu.items {
            guard let submenu = item.submenu else { continue }
            let name: MenuDiagnosticName
            if submenu.items.contains(where: { $0.view is MenuTableHostingView }) {
                name = .models
            } else if submenu.items.contains(where: { $0.view is ActivityHostingView }) {
                name = .activity
            } else {
                name = item.identifier.flatMap { MenuDiagnosticName(rawValue: $0.rawValue) } ?? .submenu
            }
            tracking.observe(submenu, name: name)
            observeSubmenus(in: submenu)
        }
    }
    func observeMainMenu(_ menu: NSMenu) { tracking.observe(menu, name: .main) }
    /// A menu popped up from a hosted view (a model's settings beside its row): not a submenu, so observed by name.
    func observe(_ menu: NSMenu, name: MenuDiagnosticName) { tracking.observe(menu, name: name) }
    func menuWillOpen(_ menu: NSMenu) { tracking.begin(menu) }
    func menuDidClose(_ menu: NSMenu) { tracking.end(menu) }
    func report(_ error: Error) { lastError = error.localizedDescription }
    var trackingRecorder: MenuTrackingRecorder { tracking }

    func recordAfterOpen(menu: NSMenu, name: MenuDiagnosticName) {
        guard !terminating else { return }
        let id = ObjectIdentifier(menu)
        guard pending[id] == nil else { return }
        pending[id] = (menu, name)
        RunLoop.main.perform(inModes: [.eventTracking, .default, .common]) { [weak self] in
            MainActor.assumeIsolated {
                guard let self, !self.terminating else { return }
                guard let entry = self.pending.removeValue(forKey: id) else { return }
                self.record(menu: entry.menu, name: entry.name)
            }
        }
    }

    /// `parentWindow`: the window a popped-up menu cascades from (it has no supermenu); `anchor`: the screen rectangle of
    /// the row it belongs beside, so its alignment can be read from the log.
    func record(
        menu: NSMenu, name: MenuDiagnosticName, view: NSView? = nil, content: NSView? = nil, scale: CGFloat = 1, parentWindow: NSWindow? = nil,
        anchor: NSRect? = nil
    ) {
        guard !terminating else { return }
        let window = view?.window ?? (menu.accessibilityWindow() as? NSWindow)
        let nativeFrame = menu.accessibilityFrame()
        // These are AppKit NSAccessibility screen coordinates, not AXUIElement/CG top-left coordinates.
        var data: [String: Any] = [
            "date": ISO8601DateFormatter().string(from: Date()), "record": "geometry", "menu": name.rawValue,
            "axFrame": Self.rect(nativeFrame), "scale": scale,
            "images": menu.items.enumerated().filter { $0.element.image != nil }.map { index, item in
                ["index": index, "preference": item.menuImagePreference, "applicable": item.applicableMenuImage as Any] as [String: Any]
            }
        ]
        for (key, value) in tracking.identifiers(menu) { data[key] = value }
        if let window {
            data["window"] = Self.rect(window.frame)
            data["windowVisible"] = window.isVisible
            data["provisionalWindow"] = window.frame.width <= 10 && window.frame.height <= 10
            let screen = window.screen ?? NSScreen.screens.first { $0.frame.intersects(window.frame) }
            data["screen"] = screen.map { Self.rect($0.frame) } ?? []
            data["screenVisible"] = screen.map { Self.rect($0.visibleFrame) } ?? []
            if let view {
                let host = view.convert(view.bounds, to: nil)
                let visible = view.convert(view.visibleRect, to: nil)
                data["host"] = Self.rect(host)
                data["hostVisible"] = Self.rect(visible)
                // The original allocation may overflow even when the fitted drawing is contained.
                data["hostOutsideWindow"] = Self.overflows(host, container: NSRect(origin: .zero, size: window.frame.size))
                data["hostClipped"] = Self.overflows(host, container: visible)
            }
            if let content {
                let frame = content.convert(content.bounds, to: nil)
                let visible = content.convert(content.visibleRect, to: nil)
                data["content"] = Self.rect(frame)
                data["contentVisible"] = Self.rect(visible)
                let windowBounds = NSRect(origin: .zero, size: window.frame.size)
                data["outsideWindow"] = Self.overflows(frame, container: windowBounds)
                data["clipped"] = Self.overflows(frame, container: visible)
            }
            if let parent = menu.supermenu {
                let p = parent.accessibilityFrame()
                data["parentAX"] = Self.rect(p)
                if let parentWindow = parent.accessibilityWindow() as? NSWindow {
                    data["side"] = Self.side(child: window.frame, parent: parentWindow.frame)
                    data["parentWindow"] = Self.rect(parentWindow.frame)
                    let overlap = window.frame.intersection(parentWindow.frame)
                    data["parentWindowOverlap"] = Self.rect(overlap)
                    data["overlapsParentWindow"] = !overlap.isEmpty && !overlap.isNull
                } else {
                    data["side"] = Self.side(child: nativeFrame, parent: p)
                }
            } else if let parentWindow {
                data["side"] = Self.side(child: window.frame, parent: parentWindow.frame)
                data["parentWindow"] = Self.rect(parentWindow.frame)
                let overlap = window.frame.intersection(parentWindow.frame)
                data["parentWindowOverlap"] = Self.rect(overlap)
                data["overlapsParentWindow"] = !overlap.isEmpty && !overlap.isNull
            } else {
                data["side"] = "root"
            }
            if let anchor {
                data["anchor"] = Self.rect(anchor)
                if let content {
                    // The first line's top against the row's top, in points (0 when level, as a cascade's first item is).
                    data["anchorOffset"] = ((window.convertToScreen(content.convert(content.bounds, to: nil)).maxY - anchor.maxY) * 100).rounded() / 100
                }
            }
        } else {
            data["geometry"] = nativeFrame.isEmpty ? "unavailable" : "accessibility-only"
            data["side"] = menu.supermenu.map { Self.side(child: nativeFrame, parent: $0.accessibilityFrame()) } ?? "root"
        }
        do { try append(data) } catch { lastError = error.localizedDescription }
    }

    static func side(child: NSRect, parent: NSRect) -> String {
        guard !child.isEmpty, !parent.isEmpty else { return "unavailable" }
        if child.midX < parent.minX { return "left" }
        if child.midX > parent.maxX { return "right" }
        return "overlap"
    }
    static func overflows(_ rect: NSRect, container: NSRect) -> Bool {
        !container.insetBy(dx: -0.5, dy: -0.5).contains(rect)
    }
    static func rect(_ r: NSRect) -> [CGFloat] {
        guard !r.isNull, !r.isInfinite else { return [] }
        return [r.minX, r.minY, r.width, r.height].map { ($0 * 100).rounded() / 100 }
    }
    func append(_ record: [String: Any]) throws {
        guard !terminating else { return }
        var line = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
        line.append(10)
        guard line.count <= Self.maximumBytes else { throw CocoaError(.fileWriteOutOfSpace) }
        // Capture/serialize on MainActor; only immutable bytes cross to the ordered background queue.
        writer.enqueue(line) { [weak self] error in self?.lastError = error }
    }

    /// Test/agent barrier, including completion delivery. Production menu callbacks never wait for disk.
    func flush() async { tracking.writeClosed(); await writer.flush() }

    /// Quit serializes already captured closes and drains records under one deadline. No fresh AppKit queries.
    /// Pending callbacks become inert in O(1); retaining their menus until exit avoids unbounded teardown here.
    @discardableResult func drainForTermination() -> Bool {
        let deadline = DispatchTime.now() + Self.terminationBudget
        tracking.terminate(until: deadline)
        terminating = true
        return writer.drain(until: deadline)
    }

}

/// The same ordered writer boundary permits deterministic termination checks without disk scheduling.
protocol MenuGeometryWriting: Sendable {
    func enqueue(_ line: Data, completion: @escaping @MainActor @Sendable (String?) -> Void)
    func flush() async
    func drain(until deadline: DispatchTime) -> Bool
}

/// No mutable shared state: queue submissions are FIFO and every file operation runs on this queue.
final class MenuGeometryWriter: MenuGeometryWriting {
    private let url: URL
    private let queue = DispatchQueue(label: "Arsenal.menu-geometry", qos: .utility)
    private let onDiskAccess: (@Sendable () -> Void)?
    init(url: URL, onDiskAccess: (@Sendable () -> Void)?) { self.url = url; self.onDiskAccess = onDiskAccess }

    func enqueue(_ line: Data, completion: @escaping @MainActor @Sendable (String?) -> Void) {
        queue.async { [self] in
            let failure: String?
            do { try persist(line); failure = nil } catch { failure = error.localizedDescription }
            DispatchQueue.main.async { completion(failure) }
        }
    }

    func flush() async {
        await withCheckedContinuation { continuation in
            queue.async { DispatchQueue.main.async { continuation.resume() } }
        }
    }

    func drain(until deadline: DispatchTime) -> Bool {
        let done = DispatchSemaphore(value: 0)
        queue.async { done.signal() }
        return done.wait(timeout: deadline) == .success
    }

    private func persist(_ line: Data) throws {
        precondition(!Thread.isMainThread, "Menu geometry disk access must stay off the main thread")
        onDiskAccess?()
        let maximumBytes = MenuDiagnostics.maximumBytes
        guard line.count <= maximumBytes else { throw CocoaError(.fileWriteOutOfSpace) }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var old = Data()
        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            let end = try handle.seekToEnd()
            // One extra byte tells us whether the retained tail starts at a record boundary.
            let readLimit = maximumBytes + 1
            let offset = end > readLimit ? end - UInt64(readLimit) : 0
            try handle.seek(toOffset: offset)
            old = try handle.read(upToCount: readLimit) ?? Data()
            if offset > 0 {
                // A legacy over-limit file can start this bounded read inside a record.
                if let newline = old.firstIndex(of: 10) { old = Data(old[old.index(after: newline)...]) } else { old = Data() }
            }
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile || error.code == .fileNoSuchFile {
            // First open: there is no log yet.
        }
        old = Self.wholeRecordsSuffix(old, maximumBytes: maximumBytes - line.count)
        old.append(line)
        try old.write(to: url, options: .atomic)
    }

    private static func wholeRecordsSuffix(_ data: Data, maximumBytes: Int) -> Data {
        guard maximumBytes > 0 else { return Data() }
        // Ignore any incomplete trailing record left by an older/external writer.
        guard let lastNewline = data.lastIndex(of: 10) else { return Data() }
        let end = data.index(after: lastNewline)
        var start = max(data.startIndex, end - maximumBytes)
        if start > data.startIndex, data[start - 1] != 10 {
            guard let newline = data[start..<end].firstIndex(of: 10) else { return Data() }
            start = data.index(after: newline)
        }
        return Data(data[start..<end])
    }
}

extension NSMenuItem {
    var menuImageIsVisible: Bool {
        if #available(macOS 27.0, *) { return menuImagePreference == 1 }
        return true
    }
    var menuImagePreference: Int {
        if #available(macOS 27.0, *) { return (value(forKey: "preferredImageVisibility") as? NSNumber)?.intValue ?? 0 }
        return 1
    }
    /// AppKit has no public effective-visibility getter. This guarded, read-only probe is diagnostics only.
    /// Missing selectors remain explicitly unknown; rendering itself always uses the visible policy.
    var applicableMenuImage: Any {
        guard image != nil else { return false }
        let selector = NSSelectorFromString("_applicableImage")
        guard responds(to: selector) else { return "unknown" }
        return perform(selector)?.takeUnretainedValue() != nil
    }
}
