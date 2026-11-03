// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

/// A soft band of light that sweeps across a row while its model downloads, is made or loads. It is a Core Animation layer animation, so it runs on the
/// render server: it moves inside a tracked menu (whose run-loop mode holds SwiftUI's timers back) and costs no draw
/// passes of the row. Present only while the row is busy, so a table at rest draws nothing; with Reduce Motion the row
/// keeps its static progress fill and no band moves.
struct BusySheen: NSViewRepresentable {
    /// Drawn on the loaded row's blue: a white band; elsewhere the label colour, fainter.
    var hot: Bool
    var cornerRadius: CGFloat

    func makeNSView(context: Context) -> SheenView { SheenView() }
    func updateNSView(_ view: SheenView, context: Context) {
        view.hot = hot
        view.cornerRadius = cornerRadius
    }
}

final class SheenView: NSView {
    static let period: CFTimeInterval = 1.8
    /// The band's width as a share of the row, and its peak opacity on a plain row and on the loaded blue.
    static let bandShare: CGFloat = 0.35, plainAlpha: CGFloat = 0.07, hotAlpha: CGFloat = 0.16
    static let animationKey = "sweep"
    /// Frame-sequence tests compare settled frames; they turn the loop off.
    nonisolated(unsafe) static var animates = true
    var hot = false { didSet { if hot != oldValue { applyColors() } } }
    var cornerRadius: CGFloat = 0 { didSet { layer?.cornerRadius = cornerRadius } }
    private let band = CAGradientLayer()

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.masksToBounds = true
        band.startPoint = CGPoint(x: 0, y: 0.5); band.endPoint = CGPoint(x: 1, y: 0.5)
        band.locations = [0, 0.5, 1]
        layer?.addSublayer(band)
        applyColors()
        // Reduce Motion switched on mid-download stops the band at once (and off starts it again).
        NSWorkspace.shared.notificationCenter.addObserver(
            self, selector: #selector(displayOptionsChanged), name: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, object: nil)
    }
    deinit { NSWorkspace.shared.notificationCenter.removeObserver(self) }
    @objc private func displayOptionsChanged() { restart() }
    required init?(coder: NSCoder) { nil }
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
    override var isOpaque: Bool { false }

    private func applyColors() {
        let base: NSColor = hot ? .white : .labelColor
        let peak = base.withAlphaComponent(hot ? Self.hotAlpha : Self.plainAlpha)
        let edge = base.withAlphaComponent(0)
        effectiveAppearance.performAsCurrentDrawingAppearance {
            band.colors = [edge.cgColor, peak.cgColor, edge.cgColor]
        }
    }
    override func viewDidChangeEffectiveAppearance() { super.viewDidChangeEffectiveAppearance(); applyColors() }

    override func layout() {
        super.layout()
        CATransaction.begin(); CATransaction.setDisableActions(true)
        let width = max(1, bounds.width * Self.bandShare)
        band.frame = CGRect(x: -width, y: 0, width: width, height: bounds.height)
        CATransaction.commit()
        restart()
    }
    override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); restart() }

    /// The sweep, from just off the leading edge to just past the trailing one, again and again.
    private func restart() {
        band.removeAnimation(forKey: Self.animationKey)
        guard Self.animates, window != nil, bounds.width > 0, !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion else { return }
        let width = band.bounds.width
        let sweep = CABasicAnimation(keyPath: "position.x")
        sweep.fromValue = -width / 2
        sweep.toValue = bounds.width + width / 2
        sweep.duration = Self.period
        sweep.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        sweep.repeatCount = .infinity
        // Every busy row sweeps in step, whenever it started.
        let now = band.convertTime(CACurrentMediaTime(), from: nil)
        sweep.beginTime = now - now.truncatingRemainder(dividingBy: Self.period)
        sweep.isRemovedOnCompletion = false
        band.add(sweep, forKey: Self.animationKey)
    }
    /// Tests: whether the band is moving.
    var sweeping: Bool { band.animation(forKey: Self.animationKey) != nil }
}
