// SPDX-License-Identifier: MIT
import AppKit
import QuartzCore
import SwiftUI
import Testing
@testable import Arsenal

/// Actual menu hosting/layers in an unordered offscreen window. Rasterization reads the presentation layer;
/// idle probes count app drawing and layout separately from the rasterizer's synchronous delegate drawing.
#if DEBUG
    @MainActor final class MotionFrames<Content: View> {
        struct Frame {
            let bitmap: NSBitmapImageRep
            let size: NSSize
            let controls: [String: NSRect]
            let layers: [String: CGRect]
            let timestamp: TimeInterval
            var bytes: Data { Data(bytes: bitmap.bitmapData!, count: bitmap.bytesPerRow * bitmap.pixelsHigh) }
            func region(_ rect: NSRect) -> Data {
                let scale = CGFloat(bitmap.pixelsWide) / size.width
                let left = max(0, Int(rect.minX * scale)), right = min(bitmap.pixelsWide, Int(rect.maxX * scale))
                let top = max(0, Int(rect.minY * scale)), bottom = min(bitmap.pixelsHigh, Int(rect.maxY * scale))
                var result = Data()
                guard left < right, top < bottom, let pointer = bitmap.bitmapData else { return result }
                for y in top..<bottom { result.append(pointer + y * bitmap.bytesPerRow + left * 4, count: (right - left) * 4) }
                return result
            }
        }
        let host: MenuContentView<Content>
        let window: NSWindow
        let container: NSView
        let lostWidth: CGFloat
        var sizes: [NSSize] = []
        var frames: [Frame] = []
        var fixedControls: [NSView] = []
        var fixedRegions: [NSRect] = []
        var animatedMeterRegions: [NSRect] = []
        private var sheenWas = true
        isolated deinit { SheenView.animates = sheenWas }
        init(host: MenuContentView<Content>, lostWidth: CGFloat = 0) {
            // A busy row's sweep loops by design (BusySheen); its frames would never settle on the expected one.
            sheenWas = SheenView.animates
            SheenView.animates = false
            self.host = host; self.lostWidth = lostWidth
            container = NSView(frame: NSRect(origin: .zero, size: host.naturalSize))
            window = NSWindow(contentRect: container.frame.offsetBy(dx: -9000, dy: -9000), styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false; window.appearance = NSAppearance(named: .darkAqua)
            window.backgroundColor = .clear; window.isOpaque = false
            window.contentView = container; container.addSubview(host)
            let original = host.onNaturalSizeChange
            host.onNaturalSizeChange = { [weak self] size in
                original?(size)
                self?.allocate()
            }
            host.onAllocationChange = { [weak self] in self?.sizes.append($0) }
            allocate()
            host.displayIfNeeded(); CATransaction.flush()
        }
        func close() { host.onAllocationChange = nil; window.contentView = nil; window.close() }
        private func allocate() {
            let size = NSSize(width: host.naturalSize.width - lostWidth, height: host.naturalSize.height)
            if window.contentLayoutRect.size != size { window.setContentSize(size) }
            if container.frame.size != size { container.setFrameSize(size) }
            host.needsLayout = true; host.layoutSubtreeIfNeeded()
        }
        func tick() async {
            // MainActor tasks and tracking/default run-loop callbacks both participate, as they do in NSMenu tracking.
            await Task.yield()
            pumpRunLoop()
            // NSMenu follows its item view size natively. The unordered fixture window needs that allocation mirrored.
            let expected = NSSize(width: host.naturalSize.width - lostWidth, height: host.naturalSize.height)
            if container.bounds.size != expected { allocate() }
            container.layoutSubtreeIfNeeded()
            host.layoutSubtreeIfNeeded()
            host.displayIfNeeded()
            CATransaction.flush()
        }
        private func pumpRunLoop() { RunLoop.main.run(until: Date(timeIntervalSinceNow: 1.0 / 120)) }
        func settle() async { for _ in 0..<12 { await tick() } }
        func expectFirstFrame(_ name: String = String(describing: Content.self)) async throws {
            let first = try capture()
            await settle()
            let settled = try capture()
            let expected = try await expectedFrame()
            #expect(first.bytes == expected.bytes && first.size == expected.size, "First frame matches independently rendered expected content")
            frames = [first, settled]
            try writeStrip("first-\(name)", layoutPasses: host.layoutPasses, resizes: sizes.count)
            #expect(first.bytes == settled.bytes && first.controls == settled.controls, "First frame must already have final content and geometry: \(first.size) → \(settled.size)")
        }
        func capture() throws -> Frame {
            #expect(!window.isVisible, "Never order a QA window onto the user's screen")
            let scale = window.backingScaleFactor
            let bitmap = try #require(
                NSBitmapImageRep(
                    bitmapDataPlanes: nil,
                    pixelsWide: Int((host.hosting.bounds.width * scale).rounded(.up)),
                    pixelsHigh: Int((host.hosting.bounds.height * scale).rounded(.up)),
                    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
            // Render the actual presentation layer. CA may invoke an AppKit drawing delegate synchronously;
            // passive idle counters are checked before rasterization, so this never counts as app invalidation.
            var controls: [String: NSRect] = [:]
            for view in allViews(host.hosting) where view.isAccessibilityElement() && view.accessibilityRole() == .button {
                let key = "\(type(of: view))/\(view.accessibilityLabel() ?? "")"
                // Include location in duplicate disambiguation; invariant controls are additionally compared by their view identity.
                controls["\(key)/\(ObjectIdentifier(view))"] = view.convert(view.bounds, to: host)
            }
            var layers: [String: CGRect] = [:]
            func walk(_ layer: CALayer) {
                layers["\(ObjectIdentifier(layer))"] = (layer.presentation() ?? layer).frame
                layer.sublayers?.forEach(walk)
            }
            if let layer = host.hosting.layer { walk(layer) }
            let layerBitmap = bitmap
            let context = try #require(NSGraphicsContext(bitmapImageRep: layerBitmap)?.cgContext)
            context.translateBy(x: 0, y: CGFloat(bitmap.pixelsHigh))
            context.scaleBy(x: CGFloat(bitmap.pixelsWide) / host.hosting.bounds.width, y: -CGFloat(bitmap.pixelsHigh) / host.hosting.bounds.height)
            let layer = try #require(host.hosting.layer)
            (layer.presentation() ?? layer).render(in: context)
            let bytes = Data(bytes: try #require(layerBitmap.bitmapData), count: layerBitmap.bytesPerRow * layerBitmap.pixelsHigh)
            let hasInk = FrameInk.hasContent(bytes)
            #expect(hasInk, "Actual CA content must render")
            let frame = Frame(
                bitmap: bitmap, size: host.frame.size, controls: controls, layers: layers,
                timestamp: ProcessInfo.processInfo.systemUptime)
            #expect(FrameInk.hasContent(frame.bytes), "No blank frame")
            for corner in [NSPoint(x: 0, y: 0), NSPoint(x: bitmap.pixelsWide - 1, y: 0)] {
                #expect((bitmap.colorAt(x: Int(corner.x), y: Int(corner.y))?.alphaComponent ?? 1) == 0, "No menu background flash")
            }
            #expect(!MenuDiagnostics.overflows(host.hosting.convert(host.hosting.bounds, to: container), container: container.bounds), "No clipped host")
            return frame
        }
        @discardableResult func transition(_ name: String, animated: Bool = false, change: () async throws -> Void) async throws -> Frame {
            let before = try capture()
            let sizeCount = sizes.count
            let layoutCount = host.layoutPasses
            let anchors = fixedControls.map { ($0, $0.convert($0.bounds, to: host)) }
            frames = [before]
            try await change()
            let count = animated ? 64 : 12
            for _ in 0..<count { await tick(); frames.append(try capture()) }
            // Render current input afresh, independently of the potentially stale/reused live host.
            let expected = try await expectedFrame()
            if animated {
                // SwiftUI's smooth duration is perceptual; its settling tail must finish independently of capture cost.
                // Keep all 64 original frames and every subsequent frame until the final pixels arrive or the deadline fails.
                let deadline = ContinuousClock.now.advanced(by: .seconds(2))
                while ContinuousClock.now < deadline && frames.last?.bytes != expected.bytes {
                    await tick(); frames.append(try capture())
                }
            }
            let after = try #require(frames.last)
            #expect(after.bytes == expected.bytes && after.size == expected.size, "\(name): final frame matches independently rendered expected content")
            #expect(MotionSequence.oneForwardSwitch(frames.map(\.size), from: before.size, to: expected.size), "\(name): allocation cannot reverse")
            #expect(
                MotionSequence.oneForwardSwitch([before.size] + Array(sizes.dropFirst(sizeCount)) + [after.size], from: before.size, to: expected.size),
                "\(name): explicit or measured allocations occur in one step without reversal")
            if !animated {
                #expect(
                    MotionSequence.oneForwardSwitch(frames.map(\.bytes), from: before.bytes, to: expected.bytes),
                    "\(name): pixels switch forward once; no flicker or incomplete/double-drawn content")
            }
            #expect(sizes.count - sizeCount <= 1, "\(name): native host grows/shrinks in one step")
            #expect(host.layoutPasses - layoutCount <= 2, "\(name): no layout cascade")
            let beforeBytes = before.bytes, afterBytes = after.bytes
            // A changing meter may interpolate, but pixels which agree at both endpoints must remain untouched.
            // Check every RGBA pixel in the common image area, including text, backgrounds and other meters.
            let invariantPixels: [Int] =
                animated && before.bitmap.bytesPerRow == after.bitmap.bytesPerRow
                ? beforeBytes.withUnsafeBytes { a in
                    afterBytes.withUnsafeBytes { b in
                        stride(from: 0, to: min(a.count, b.count), by: 4).filter { i in
                            let scale = CGFloat(before.bitmap.pixelsWide) / before.size.width
                            let point = NSPoint(x: CGFloat((i % before.bitmap.bytesPerRow) / 4) / scale, y: CGFloat(i / before.bitmap.bytesPerRow) / scale)
                            return !animatedMeterRegions.contains { $0.contains(point) }
                                && a[i] == b[i] && a[i + 1] == b[i + 1] && a[i + 2] == b[i + 2] && a[i + 3] == b[i + 3]
                        }
                    }
                } : []
            // Controls which survive in the same positions at both endpoints cannot take a detour on an intermediate frame.
            for (key, rect) in before.controls where after.controls[key] == rect {
                for frame in frames { #expect(frame.controls[key] == rect, "\(name): stable neighbour \(key)") }
            }
            for frame in frames {
                if animated {
                    let bytes = frame.bytes
                    let unchanged = bytes.withUnsafeBytes { current in
                        beforeBytes.withUnsafeBytes { original in
                            invariantPixels.allSatisfy { i in
                                i + 3 < current.count && current[i] == original[i] && current[i + 1] == original[i + 1]
                                    && current[i + 2] == original[i + 2] && current[i + 3] == original[i + 3]
                            }
                        }
                    }
                    #expect(unchanged, "\(name): animation touches only pixels which changed")
                }
                for rect in fixedRegions { #expect(frame.region(rect) == before.region(rect), "\(name): unchanged region stays identical") }
                for (view, rect) in anchors {
                    let key = "\(type(of: view))/\(view.accessibilityLabel() ?? "")/\(ObjectIdentifier(view))"
                    #expect(frame.controls[key] == rect, "\(name): unaffected control keeps exact frame and identity")
                }
                // A native button can never exist twice at the same frame: catches transient duplicate representables.
                let rects = Array(frame.controls.values)
                #expect(Set(rects.map { NSStringFromRect($0) }).count == rects.count, "\(name): double-drawn button")
            }
            if animated {
                for rect in animatedMeterRegions {
                    // Meter fill is opaque, its track translucent. Its occupied pixels must progress monotonically,
                    // including the tiny capsule's changing antialiased cap when the first sample arrives.
                    let counts = frames.map { frame in
                        frame.region(rect).withUnsafeBytes { raw in stride(from: 3, to: raw.count, by: 4).filter { raw[$0] >= 128 }.count }
                    }
                    #expect(counts == (counts.first ?? 0 <= counts.last ?? 0 ? counts.sorted() : counts.sorted(by: >)), "\(name): meter cannot flash/overshoot its data")
                }
            }
            try writeStrip(name, layoutPasses: host.layoutPasses - layoutCount, resizes: sizes.count - sizeCount)
            return after
        }
        private func expectedFrame() async throws -> Frame {
            // The reference is the final state, independent of CPU work incidentally finishing its entrance animation.
            let reference = MenuContentView(
                rootView: host.hosting.rootView.content.transaction {
                    $0.disablesAnimations = true; $0.animation = nil
                })
            reference.setNaturalSize(reference.contentFittingSize())
            let expected = MotionFrames<_>(host: reference, lostWidth: lostWidth)
            defer { expected.close() }
            await expected.settle()
            let originalActions = allViews(host).compactMap { $0 as? RowActionView }
            let expectedActions = allViews(reference).compactMap { $0 as? RowActionView }
            #expect(originalActions.count == expectedActions.count)
            for (original, copy) in zip(originalActions, expectedActions) { copy.setHovered(original.hovered) }
            if let original = MenuTests.find(RefreshControlView.self, in: host), let copy = MenuTests.find(RefreshControlView.self, in: reference) {
                copy.forcedHover = original.color == ArsenalPalette.onMenu.t1 && original.enabled
                copy.needsDisplay = true
            }
            reference.displayIfNeeded(); CATransaction.flush()
            let snapshot = try expected.capture()
            return Frame(
                bitmap: snapshot.bitmap, size: snapshot.size, controls: snapshot.controls, layers: snapshot.layers,
                timestamp: snapshot.timestamp)
        }
        func expectIdle(_ name: String) async throws {
            await settle()
            let first = try capture()
            let layouts = host.layoutPasses, hostingLayouts = host.hostingLayoutPasses, draws = host.hostingDrawPasses
            let nativeDraws = drawCounts()
            for _ in 0..<12 { await tick() }
            #expect(host.layoutPasses == layouts && host.hostingLayoutPasses == hostingLayouts, "\(name): idle layout")
            #expect(host.hostingDrawPasses == draws, "\(name): idle drawing (capture excluded)")
            #expect(nativeDraws == drawCounts(), "\(name): native controls idle drawing")
            // Sample consecutive frames too, but exclude each capture's synchronous drawing from the next interval.
            for _ in 0..<12 {
                let sampledDraws = host.hostingDrawPasses, sampledNativeDraws = drawCounts()
                await tick()
                #expect(host.hostingDrawPasses == sampledDraws && drawCounts() == sampledNativeDraws, "\(name): no drawing between captures")
                let frame = try capture()
                #expect(first.bytes == frame.bytes && first.controls == frame.controls && first.layers == frame.layers, "\(name): every idle frame/layer stays identical")
            }
        }
        private func drawCounts() -> [ObjectIdentifier: Int] {
            var result: [ObjectIdentifier: Int] = [:]
            for view in allViews(host) {
                let count: Int?
                switch view {
                case let v as RowActionView: count = v.drawPasses
                case let v as GearView: count = v.drawPasses
                case let v as RefreshControlView: count = v.drawPasses
                case let v as PinnedRowView: count = v.drawPasses
                case let v as GlyphButton: count = v.drawPasses
                case let v as ActionButtonView: count = v.drawPasses
                case let v as MenuSegments: count = v.drawPasses
                case let v as MenuSwitch: count = v.drawPasses
                case let v as GroupHeaderView: count = v.drawPasses
                case let v as MenuField: count = v.drawPasses
                case let v as MenuSlider: count = v.drawPasses
                case let v as TierSegments.Control: count = v.drawPasses
                case let v as PrecisionLinesView: count = v.drawPasses
                default: count = nil
                }
                if let count { result[ObjectIdentifier(view)] = count }
            }
            return result
        }
        private func writeStrip(_ name: String, layoutPasses: Int, resizes: Int) throws {
            guard let output = ProcessInfo.processInfo.environment["ARSENAL_MOTION_STRIPS"] else { return }
            let directory = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            // Every sampled frame is retained at full resolution; a compact contact strip is also written for review.
            for (index, frame) in frames.enumerated() {
                try frame.bitmap.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("\(name)-\(index).png"))
            }
            let selected = [frames[0], frames[1], frames[frames.count / 2], frames[frames.count - 1]]
            let width = selected.map { $0.bitmap.pixelsWide }.max() ?? 1
            let height = selected.map { $0.bitmap.pixelsHigh }.max() ?? 1
            let strip = try #require(
                NSBitmapImageRep(
                    bitmapDataPlanes: nil, pixelsWide: width * 4, pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4,
                    hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
            NSGraphicsContext.saveGraphicsState(); NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: strip)
            for (index, frame) in selected.enumerated() {
                let image = NSImage(size: NSSize(width: frame.bitmap.pixelsWide, height: frame.bitmap.pixelsHigh)); image.addRepresentation(frame.bitmap)
                image.draw(in: NSRect(x: index * width, y: height - frame.bitmap.pixelsHigh, width: frame.bitmap.pixelsWide, height: frame.bitmap.pixelsHigh))
            }
            NSGraphicsContext.restoreGraphicsState()
            try strip.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("\(name)-strip.png"))
            let intervals = zip(frames, frames.dropFirst()).map { $1.timestamp - $0.timestamp }
            let summary = "frames=\(frames.count) actual_intervals_s=\(intervals) layouts=\(layoutPasses) resizes=\(resizes)\n"
            try summary.write(to: directory.appendingPathComponent("\(name).txt"), atomically: true, encoding: .utf8)
        }
    }

    @MainActor func allViews(_ root: NSView) -> [NSView] { [root] + root.subviews.flatMap(allViews) }

#endif

#if DEBUG
    /// Endpoint correctness is supplied by the caller; the last sample is never used as its own oracle.
    enum MotionSequence {
        static func oneForwardSwitch<Value: Equatable>(_ samples: [Value], from old: Value, to expected: Value) -> Bool {
            guard samples.first == old, samples.last == expected else { return false }
            var reachedExpected = old == expected
            for sample in samples {
                if sample == expected { reachedExpected = true } else if sample != old || reachedExpected { return false }
            }
            return true
        }
    }
#endif

/// The original sampled-byte predicate: more than eight distinct values at stride 31.
/// Nine distinct values establish the predicate; later samples cannot change that result.
enum FrameInk {
    static func hasContent(_ bytes: Data) -> Bool {
        bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var seen = [Bool](repeating: false, count: 256)
            var count = 0
            for index in stride(from: 0, to: raw.count, by: 31) {
                let value = Int(raw[index])
                if !seen[value] {
                    seen[value] = true; count += 1
                    if count > 8 { return true }
                }
            }
            return false
        }
    }
}

@Test func frameInkKeepsTheOriginalSampleAndThreshold() {
    for distinct in 0...12 {
        var data = Data(repeating: 0, count: 31 * 32)
        for index in 0..<distinct { data[index * 31] = UInt8(index) }
        let original = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Set(stride(from: 0, to: raw.count, by: 31).map { raw[$0] }).count > 8 }
        #expect(FrameInk.hasContent(data) == original)
    }
    #expect(!FrameInk.hasContent(Data()))
    #expect(!FrameInk.hasContent(Data(repeating: 255, count: 4096)))
    // Diversity outside the original stride must not make a blank sampled frame pass.
    var unsampled = Data(repeating: 0, count: 128)
    for index in 1...12 { unsampled[index] = UInt8(index) }
    #expect(!FrameInk.hasContent(unsampled))
}
