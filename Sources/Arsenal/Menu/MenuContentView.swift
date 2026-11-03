// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

/// AppKit owns the allocation; SwiftUI owns the fit transform so its controls and hit testing agree with drawing.
class MenuContentView<Content: View>: NSView {
    #if DEBUG
        private(set) var layoutPasses = 0
        var onAllocationChange: ((NSSize) -> Void)?
        var hostingLayoutPasses: Int { (hosting as? VibrantHostingView)?.layoutPasses ?? 0 }
        var hostingDrawPasses: Int { (hosting as? VibrantHostingView)?.drawPasses ?? 0 }
    #endif
    let hosting: NSHostingView<FittedMenuContent<Content>>
    private let content: Content
    private(set) var naturalSize: NSSize = .zero
    private(set) var contentScale: CGFloat = 1
    var onOpenLayout: ((MenuContentView) -> Void)?
    var onNaturalSizeChange: ((NSSize) -> Void)?
    private let tracksContentSize: Bool
    private var measuredSize: NSSize?
    private var measuring = false
    private var checking = false
    override var allowsVibrancy: Bool { true }
    override var isFlipped: Bool { true }
    override var intrinsicContentSize: NSSize { naturalSize }

    init(rootView: Content, tracksContentSize: Bool = false) {
        content = rootView
        self.tracksContentSize = tracksContentSize
        hosting = VibrantHostingView(rootView: FittedMenuContent(content: rootView, size: .zero, scale: 1))
        super.init(frame: .zero)
        hosting.sizingOptions = []
        hosting.wantsLayer = true
        hosting.layer?.backgroundColor = NSColor.clear.cgColor
        hosting.layer?.isOpaque = false
        addSubview(hosting)
    }
    required init?(coder: NSCoder) { nil }
    /// Measure the original content, not the fitted host's constrained/intrinsic frame.
    func contentFittingSize() -> NSSize { NSHostingView(rootView: content).fittingSize }
    func setNaturalSize(_ size: NSSize) {
        // An explicit current allocation supersedes any queued geometry measurement from an older root layout.
        measuredSize = size
        guard size != naturalSize else { return }
        naturalSize = size
        invalidateIntrinsicContentSize()
        setFrameSize(size)
    }
    /// Geometry is measured before the fit transform, at the content's ideal size. Changing the native
    /// allocation therefore cannot change the measurement and start a resize/measure feedback loop.
    private func contentSizeChanged(_ size: NSSize) {
        guard size.width.isFinite, size.height.isFinite, size.width > 0, size.height > 0 else { return }
        measuredSize = size
        guard !measuring else { return }
        measuring = true
        // Leave SwiftUI's layout pass before mutating its AppKit container; also works during menu tracking.
        RunLoop.main.perform(inModes: [.eventTracking, .default, .common]) { [weak self] in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.measuring = false
                guard let size = self.measuredSize, size != self.naturalSize else { return }
                self.setNaturalSize(size)
                self.onNaturalSizeChange?(size)
                self.recordAfterLayout()
            }
        }
    }
    override func setFrameSize(_ newSize: NSSize) {
        #if DEBUG
            let changed = frame.size != newSize
        #endif
        super.setFrameSize(newSize)
        #if DEBUG
            if changed { onAllocationChange?(newSize) }
        #endif
        needsLayout = true
    }
    /// A custom item may retain its requested width beyond its native viewer. Check every ancestor and the window.
    func availableRect() -> NSRect {
        var available = bounds
        var ancestor = superview
        while let view = ancestor {
            available = available.intersection(convert(view.bounds, from: view))
            ancestor = view.superview
        }
        if let window {
            available = available.intersection(convert(NSRect(origin: .zero, size: window.frame.size), from: nil))
        }
        return available.isNull ? .zero : available
    }
    override func layout() {
        #if DEBUG
            layoutPasses += 1
        #endif
        super.layout()
        guard naturalSize.width > 0, naturalSize.height > 0 else { return }
        // A submenu contains this host alone, but AppKit may displace its item viewer rather
        // than the host itself. Repair the outermost displaced allocation below the window's
        // content view. Moving only this child leaves it outside its item-sized ancestors.
        let wasDisplaced = window != nil && convert(bounds, to: nil).minY < -0.5
        if let window, wasDisplaced, naturalSize.height <= window.contentLayoutRect.height {
            onOpenLayout?(self) // Log the captured native allocation before repair.
            var allocation: NSView = self
            var ancestor = superview
            while let view = ancestor, view !== window.contentView {
                let rect = view.convert(view.bounds, to: nil)
                if rect.minY < -0.5, rect.height <= window.contentLayoutRect.height { allocation = view }
                ancestor = view.superview
            }
            if let parent = allocation.superview {
                let current = allocation.convert(allocation.bounds, to: nil)
                let bottom = (window.contentLayoutRect.height - current.height) / 2
                let delta = (bottom - current.minY) * (parent.isFlipped ? -1 : 1)
                allocation.setFrameOrigin(NSPoint(x: allocation.frame.minX, y: allocation.frame.minY + delta))
            }
        }
        let room = availableRect()
        // AppKit temporarily gives a detached/empty menu a 10×10 window. That is not a usable
        // allocation: publishing its fit would replace a live SwiftUI tree with a 0.01-scale tree.
        // Keep the last complete frame until the native viewer has its real allocation.
        if let window, window.frame.width <= 10, window.frame.height <= 10 {
            if hosting.frame.isEmpty {
                hosting.frame = NSRect(origin: .zero, size: naturalSize)
                hosting.rootView = FittedMenuContent(
                    content: content, size: naturalSize, scale: 1,
                    onSizeChange: tracksContentSize ? { [weak self] in self?.contentSizeChanged($0) } : nil)
            }
            onOpenLayout?(self)
            return
        }
        guard !room.isEmpty else { onOpenLayout?(self); return }
        // A requested height change precedes the native viewer's reallocation. Fitting that
        // ordinary growth to the old height would briefly rebuild controls at a smaller scale.
        // A displaced viewer (the logged negative y case) does need vertical containment.
        let displaced = window != nil && convert(bounds, to: nil).minY < -0.5
        let heightScale = displaced || wasDisplaced ? room.height / naturalSize.height : 1
        let scale = max(0, min(1, room.width / naturalSize.width, heightScale))
        let size = NSSize(width: naturalSize.width * scale, height: naturalSize.height * scale)
        if hosting.rootView.size != naturalSize || contentScale != scale {
            hosting.rootView = FittedMenuContent(
                content: content, size: naturalSize, scale: scale,
                onSizeChange: tracksContentSize ? { [weak self] in self?.contentSizeChanged($0) } : nil)
        }
        contentScale = scale
        hosting.frame = NSRect(origin: room.origin, size: size)
        hosting.layoutSubtreeIfNeeded()
        // Every use of the safety net leaves evidence, even on a resize after the menu opened.
        if scale < 1 { onOpenLayout?(self) }
    }
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        NotificationCenter.default.removeObserver(self, name: NSWindow.didChangeOcclusionStateNotification, object: nil)
        if let window {
            NotificationCenter.default.addObserver(self, selector: #selector(windowVisibilityChanged), name: NSWindow.didChangeOcclusionStateNotification, object: window)
        }
        recordAfterLayout()
    }
    @objc private func windowVisibilityChanged(_ note: Notification) { recordAfterLayout() }

    /// One tracking-mode block on attachment/open; no timer, idle sampler or UI.
    func recordAfterLayout() {
        guard !checking else { return }
        checking = true
        RunLoop.main.perform(inModes: [.eventTracking, .default, .common]) { [weak self] in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.checking = false
                guard self.window?.isVisible == true else { return }
                self.needsLayout = true
                self.layoutSubtreeIfNeeded()
                if self.contentScale == 1 { self.onOpenLayout?(self) }
            }
        }
    }
}
struct FittedMenuContent<Content: View>: View {
    let content: Content
    let size: NSSize
    let scale: CGFloat
    var onSizeChange: ((NSSize) -> Void)?
    @ViewBuilder private var measuredContent: some View {
        if let onSizeChange {
            content.fixedSize(horizontal: true, vertical: true)
                .onGeometryChange(for: NSSize.self) {
                    $0.size
                } action: {
                    onSizeChange($0)
                }
        } else {
            content
        }
    }
    var body: some View {
        if scale == 1 {
            measuredContent
        } else {
            measuredContent.frame(width: size.width, height: size.height, alignment: .topLeading)
                .scaleEffect(scale, anchor: .topLeading)
                .frame(width: size.width * scale, height: size.height * scale, alignment: .topLeading)
        }
    }
}
private final class VibrantHostingView<Content: View>: NSHostingView<Content> {
    override var allowsVibrancy: Bool { true }
    #if DEBUG
        private(set) var layoutPasses = 0
        private(set) var drawPasses = 0
        override func layout() { layoutPasses += 1; super.layout() }
        override func draw(_ dirtyRect: NSRect) { drawPasses += 1; super.draw(dirtyRect) }
    #endif
}
