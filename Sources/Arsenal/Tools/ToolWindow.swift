// SPDX-License-Identifier: MIT
import AppKit
import Combine
import SwiftUI

// Every window Arsenal opens besides Get more models — chat, image, operation, a model's settings — in the model
// browser's style: no visible title bar, the content's own top line sits level with the
// traffic lights, the palette's canvas, the app's appearance (dark, light, translucent). One window per model and tool,
// kept while the model stays loaded; closing it keeps the conversation.

/// Command keys for a menu-bar app, which has no main menu to carry them: ⌘W closes; the edit keys reach the focused
/// text view (without an Edit menu, ⌘V would do nothing in a text field).
final class ToolWindow: NSWindow {
    /// Offscreen renders and tests place the window far off every screen; AppKit would otherwise pull a titled window
    /// back until a corner shows on the display. Set, every tool window is also built there, so one shown through the
    /// real path (`ToolWindowController.show`) is never seen.
    nonisolated(unsafe) static var keepOffscreenFrames = false
    static let offscreenOrigin = NSPoint(x: -9000, y: -9000)
    override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect {
        Self.keepOffscreenFrames ? frameRect : super.constrainFrameRect(frameRect, to: screen)
    }
    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        guard let key = event.charactersIgnoringModifiers?.lowercased() else { return super.performKeyEquivalent(with: event) }
        let action: Selector?
        switch (key, flags) {
        case ("w", .command): performClose(nil); return true
        case ("x", .command): action = #selector(NSText.cut(_:))
        case ("c", .command): action = #selector(NSText.copy(_:))
        case ("v", .command): action = #selector(NSText.paste(_:))
        case ("a", .command): action = #selector(NSText.selectAll(_:))
        case ("z", .command): action = Selector(("undo:"))
        case ("z", [.command, .shift]): action = Selector(("redo:"))
        default: action = nil
        }
        if let action, NSApp.sendAction(action, to: nil, from: self) { return true }
        return super.performKeyEquivalent(with: event)
    }
}

/// A vibrant backdrop under the palette's tint (translucent appearance only).
struct VisualEffectBackdrop: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .underWindowBackground; v.blendingMode = .behindWindow; v.state = .active
        return v
    }
    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}

/// The window surface every tool draws on: canvas (or the translucent tint over the desktop), palette, colour scheme.
struct ToolSurface<Content: View>: View {
    @ObservedObject var appearance = Appearance.shared
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View {
        let p = appearance.palette
        // The content's own top line sits in the title bar's band, level with the traffic lights.
        content
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .ignoresSafeArea(.container, edges: .top)
            .background {
                ZStack {
                    if p.translucent { VisualEffectBackdrop() }
                    p.canvas.swiftUI
                }.ignoresSafeArea()
            }
            .foregroundStyle(p.t1.swiftUI)
            .environment(\.colorScheme, appearance.colorScheme)
            .environment(\.palette, p)
    }
}

private struct PaletteKey: EnvironmentKey { static let defaultValue = ArsenalPalette.dark }
extension EnvironmentValues {
    var palette: ArsenalPalette {
        get { self[PaletteKey.self] }
        set { self[PaletteKey.self] = newValue }
    }
}

@MainActor final class ToolWindowController: NSWindowController, NSWindowDelegate {
    private var observers: Set<AnyCancellable> = []
    var onClose: () -> Void = {}
    /// This window's key in ControlFrames.
    private(set) var scope = UUID()

    /// The title bar's height on macOS 26: the band the tools' top line is centred in.
    static let band: CGFloat = 32

    /// `resizable` false: a window sized by its content (settings), which the user does not resize.
    /// A frame saved before the rename (under the legacy prefix) becomes this window's frame once, when it has none.
    static func adoptLegacyFrame(_ autosave: String, defaults: UserDefaults = .standard) {
        guard autosave.hasPrefix("Arsenal") else { return }
        let key = "NSWindow Frame " + autosave, legacy = "NSWindow Frame Mobius" + autosave.dropFirst("Arsenal".count) // legacy-compat: frames saved before the rename
        guard defaults.object(forKey: key) == nil, let frame = defaults.string(forKey: legacy) else { return }
        defaults.set(frame, forKey: key)
    }
    init<Content: View>(title: String, autosave: String, size: NSSize, minSize: NSSize, content: Content, resizable: Bool = true) {
        let window = ToolWindow(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: NSWindow.StyleMask([.titled, .closable, .miniaturizable, .fullSizeContentView]).union(resizable ? .resizable : []), backing: .buffered,
            defer: false)
        window.title = title
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.minSize = minSize
        window.isReleasedWhenClosed = false
        window.isMovableByWindowBackground = false
        let scope = UUID()
        self.scope = scope
        let host = NSHostingView(rootView: ToolSurface { content.environment(\.controlScope, scope) })
        host.sizingOptions = []
        window.contentView = host
        super.init(window: window)
        window.delegate = self
        window.center()
        Self.adoptLegacyFrame(autosave)
        window.setFrameAutosaveName(autosave)
        if ToolWindow.keepOffscreenFrames { window.setFrameOrigin(ToolWindow.offscreenOrigin) }
        Appearance.shared.$mode.sink { [weak self] _ in DispatchQueue.main.async { self?.applyAppearance() } }.store(in: &observers)
        applyAppearance()
    }
    required init?(coder: NSCoder) { nil }

    func applyAppearance() {
        guard let window else { return }
        let a = Appearance.shared, p = a.palette
        window.appearance = a.nsAppearance
        window.isOpaque = !p.translucent
        window.backgroundColor = p.translucent ? .clear : p.canvas
    }

    /// Brings the window front as Get more models does (BrowserWindowController.show): a menu-bar app is not active
    /// on its own, so it activates first.
    func show() {
        showWindow(nil)
        Self.activateApp()
        window?.makeKeyAndOrderFront(nil)
    }
    /// Makes Arsenal the active app. Tests count it instead: activating the test runner would take the keyboard from
    /// whatever app is in front.
    static var activateApp: () -> Void = { NSApp.activate(ignoringOtherApps: true) }
    func windowWillClose(_ notification: Notification) {
        onClose()
        // The last tool window closing lets go of every picture, frame and page the windows decoded.
        let closing = window
        let open = NSApp.windows.contains { $0 !== closing && $0.delegate is ToolWindowController && ($0.isVisible || $0.isMiniaturized) }
        if !open { MediaCaches.clear() }
    }

    /// ⌘V that no text view in the window takes (a window that accepts a pasted image): ToolWindow sends `paste:` up
    /// the responder chain, which ends at this window's delegate. Unset, the controller does not answer `paste:`.
    var onPaste: ((NSPasteboard) -> Bool)? { didSet { pastes = onPaste != nil } }
    nonisolated(unsafe) private var pastes = false
    @objc func paste(_ sender: Any?) { _ = onPaste?(NSPasteboard.general) }
    override func responds(to aSelector: Selector!) -> Bool {
        aSelector == #selector(paste(_:)) ? pastes : super.responds(to: aSelector)
    }
}

/// Where each named control was laid out, for the hit-testing tests (they click the real event path at its centre). Off
/// in the app: the modifier then adds nothing.
@MainActor enum ControlFrames {
    static var recording = false
    /// SwiftUI global frames (window coordinates, top-left origin) by window, then by control name.
    static var frames: [UUID: [String: CGRect]] = [:]
    /// Window hovers (`windowHelp`): global frames by window, then by hover text.
    static var helps: [UUID: [String: CGRect]] = [:]
}

private struct ControlScopeKey: EnvironmentKey { static let defaultValue: UUID? = nil }
extension EnvironmentValues {
    /// The tool window a control belongs to (for ControlFrames).
    var controlScope: UUID? {
        get { self[ControlScopeKey.self] }
        set { self[ControlScopeKey.self] = newValue }
    }
}

private struct ControlFrameRecorder: ViewModifier {
    let id: String
    @Environment(\.controlScope) private var scope
    func body(content: Content) -> some View {
        if ControlFrames.recording, let scope {
            content.background(
                GeometryReader { g in
                    Color.clear
                        .onAppear { ControlFrames.frames[scope, default: [:]][id] = g.frame(in: .global) }
                        .onChange(of: g.frame(in: .global)) { _, frame in ControlFrames.frames[scope, default: [:]][id] = frame }
                })
        } else {
            content
        }
    }
}

extension View {
    /// Names a control for the hit-testing tests (no effect in the app).
    func controlID(_ id: String) -> some View { modifier(ControlFrameRecorder(id: id)) }
    /// Hover text in a window: SwiftUI's own `.help`, which the window's SwiftUI host shows wherever AppKit's hit test
    /// reaches that host (a button keeps its own tooltip; nothing is laid over it). Recorded for the hover tests
    /// (HoverReachTests: the host must be the view hit at its centre). In the Models menu SwiftUI help never shows; menu
    /// content uses TooltipCell's `cellTooltip` instead.
    func windowHelp(_ text: String?) -> some View { modifier(HelpRecorder(text: text)) }
}

private struct HelpRecorder: ViewModifier {
    let text: String?
    @Environment(\.controlScope) private var scope
    func body(content: Content) -> some View {
        let text = self.text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if text.isEmpty {
            content
        } else if ControlFrames.recording, let scope {
            content.help(text).background(
                GeometryReader { g in
                    Color.clear
                        .onAppear { ControlFrames.helps[scope, default: [:]][text] = g.frame(in: .global) }
                        .onChange(of: g.frame(in: .global)) { _, frame in ControlFrames.helps[scope, default: [:]][text] = frame }
                })
        } else {
            content.help(text)
        }
    }
}
