// SPDX-License-Identifier: MIT
import AppKit
import CryptoKit
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalCore

/// The release reviews' app findings, each with the check that keeps
/// it fixed. Nothing is shown on screen (windows stay unordered, far off any screen); no model runs.
@MainActor @Suite(.menuState, .serialized) final class GoNoGoAppTests {
    init() {
        _ = NSApplication.shared
        ToolWindow.keepOffscreenFrames = true
        SoundPlayer.silent = true
    }
    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-gonogo-\(UUID().uuidString)", isDirectory: true)
    deinit { try? FileManager.default.removeItem(at: scratch) }
    private static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    // MARK: A6. Pictures are cached at display size

    /// Two 24 MP pictures (a before and its after) each decoded at the screen's size, so both stay in the cache and a
    /// redraw never rereads either file; the copy keeps the picture's aspect.
    @Test func twentyFourMegapixelPicturesStayCachedSideBySide() throws {
        let before = try CapabilityFixture.photo(scratch.appendingPathComponent("before.jpg"), width: 6000, height: 4000)
        let after = try CapabilityFixture.photo(scratch.appendingPathComponent("after.png"), width: 6000, height: 4000)
        MediaCaches.clear()
        defer { MediaCaches.clear() }
        let screen = MediaCaches.displayPixels
        let a = try #require(Thumbnail.image(before)), b = try #require(Thumbnail.image(after))
        #expect(Thumbnail.cache.count == 2 && Thumbnail.cache.bytes <= Thumbnail.cache.limit)
        // The first one is still the same decoded copy after the second came in: no eviction, no reread.
        #expect(Thumbnail.image(before) === a && Thumbnail.image(after) === b)
        for image in [a, b] {
            #expect(abs(image.size.width / image.size.height - 1.5) < 0.01)
            let rep = try #require(image.representations.first)
            #expect(rep.pixelsWide <= Int(screen.width) && rep.pixelsHigh <= Int(screen.height), "\(rep.pixelsWide)×\(rep.pixelsHigh) for \(screen)")
            // Fitted to the screen, not shrunk below it: the side that limits it fills the screen's.
            #expect(rep.pixelsWide == Int(screen.width) || rep.pixelsHigh == Int(screen.height) || rep.pixelsWide == 6000)
            #expect(abs(Double(rep.pixelsWide) / Double(rep.pixelsHigh) - 1.5) < 0.01)
        }
        // The full bitmaps (96 MB each) would not both fit the 128 MB the cache had before.
        #expect(MediaCaches.bytes(a) + MediaCaches.bytes(b) < 2 * 6000 * 4000 * 4 / 2)
    }

    /// A picture smaller than the screen is decoded whole, never scaled up; fitting keeps the aspect.
    @Test func displayCopiesNeverScaleUp() throws {
        #expect(MediaCaches.fitted(CGSize(width: 640, height: 480), in: CGSize(width: 3456, height: 2234)) == CGSize(width: 640, height: 480))
        #expect(MediaCaches.fitted(CGSize(width: 6000, height: 4000), in: CGSize(width: 3456, height: 2234)) == CGSize(width: 3351, height: 2234))
        #expect(MediaCaches.fitted(CGSize(width: 4000, height: 6000), in: CGSize(width: 3456, height: 2234)) == CGSize(width: 1489, height: 2234))
        let small = try CapabilityFixture.photo(scratch.appendingPathComponent("small.jpg"), width: 640, height: 480)
        let image = try #require(MediaCaches.displayImage(small, fitting: CGSize(width: 3456, height: 2234)))
        #expect(image.representations.first?.pixelsWide == 640 && image.size == NSSize(width: 640, height: 480), "\(image.representations) \(image.size)")
    }

    /// The other pictures decoded per file in the tool windows (Select's mask tints, Depth's coloured map) are kept at
    /// display size too; Depth's near and far still come from every pixel.
    @Test func masksAndDepthMapsAreKeptAtDisplaySize() throws {
        MediaCaches.displayOverride = CGSize(width: 800, height: 600)
        defer { MediaCaches.displayOverride = nil; MediaCaches.clear() }
        let mask = scratch.appendingPathComponent("mask.png")
        try Self.grey(mask, width: 2400, height: 1800, bits: 8) { x, y in x > 1200 && y > 900 ? 255 : 0 }
        let alpha = try #require(MaskOverlay.alpha(mask))
        let alphaRep = try #require(alpha.representations.first)
        #expect(alphaRep.pixelsWide <= 800 && alphaRep.pixelsHigh <= 600 && alphaRep.pixelsWide >= 799, "\(alphaRep.pixelsWide)×\(alphaRep.pixelsHigh)")
        #expect(MaskOverlay.alpha(mask) === alpha)
        #expect(try #require(MaskOverlay.alpha(mask, edge: true)).representations.first.map { $0.pixelsWide <= 800 } == true)

        // One far pixel and one near pixel in a 2400 × 1800 map: a display copy would average them away, the range keeps them.
        let depth = scratch.appendingPathComponent("depth.png")
        try Self.grey(depth, width: 2400, height: 1800, bits: 16) { x, y in x == 7 && y == 9 ? 65535 : x == 2000 && y == 1500 ? 0 : 30000 }
        let rendered = try #require(DepthImage.render(depth, colour: true, nearIsHigh: true))
        let depthRep = try #require(rendered.image.representations.first)
        #expect(depthRep.pixelsWide <= 800 && depthRep.pixelsHigh <= 600)
        #expect(abs(rendered.image.size.width / rendered.image.size.height - 4.0 / 3) < 0.01)
        #expect(rendered.low == 0 && rendered.high == 1)
    }

    /// A grey PNG, 8 or 16 bits, from a value per pixel (x from the left, y from the top).
    static func grey(_ url: URL, width: Int, height: Int, bits: Int, value: (Int, Int) -> Int) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let bytes = bits / 8
        var data = [UInt8](repeating: 0, count: width * height * bytes)
        for y in 0..<height {
            for x in 0..<width {
                let v = value(x, y), i = (y * width + x) * bytes
                if bytes == 2 { data[i] = UInt8(v >> 8); data[i + 1] = UInt8(v & 0xff) } else { data[i] = UInt8(v) }
            }
        }
        let provider = try #require(CGDataProvider(data: Data(data) as CFData))
        let image = try #require(
            CGImage(
                width: width, height: height, bitsPerComponent: bits, bitsPerPixel: bits, bytesPerRow: width * bytes, space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | (bits == 16 ? CGBitmapInfo.byteOrder16Big.rawValue : 0)), provider: provider,
                decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let dest = try #require(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(dest, image, nil)
        #expect(CGImageDestinationFinalize(dest))
    }

    // MARK: A8. A saved option the choices no longer list

    /// Make 3D's detail and texture size were free whole numbers; a saved 768 or 3000 is in none of TRELLIS's choices
    /// now. The picker starts at the declared default (1024) instead of a blank, and the request sends that default.
    @Test func aSavedOptionOutsideTheChoicesStartsAtTheDefault() throws {
        let session = try make3D(settings: ["resolution": .number(768), "texture_size": .number(3000), "steps": .number(12)])
        let form = try #require(session.form)
        for (key, value) in [("resolution", 1024.0), ("texture_size", 1024.0)] {
            let field = try #require(form.fields.first { $0.key == key })
            guard case .generic(.choice(let choices)) = field.control else { Issue.record("\(key) is not a choice: \(field.control)"); continue }
            let shown = FieldControl.choiceIndex(session.value(key), choices: choices, auto: false, defaultValue: field.defaultValue)
            #expect(choices.indices.contains(shown) && choices[shown] == .number(value), "\(key): \(session.value(key))")
            #expect(session.value(key) == .choice(choices.firstIndex(of: .number(value))))
        }
        // A saved value the choices list is kept, and the request carries what the pickers show.
        #expect(try make3D(settings: ["resolution": .number(1536)]).value("resolution") == .choice(2))
        session.set("image", .files([try CapabilityFixture.photo(scratch.appendingPathComponent("3d.jpg"), width: 64, height: 64)]))
        let input = try form.input(session.values, seed: nil) { $0.path }
        #expect(input["resolution"] == .number(1024) && input["texture_size"] == .number(1024))
    }

    /// The picker's selection for every value a choice can hold.
    @Test func aChoicePickerNeverSelectsNothing() {
        let choices: [JSONValue] = [.number(512), .number(1024), .number(1536)]
        #expect(FieldControl.choiceIndex(.choice(2), choices: choices, auto: false, defaultValue: .number(1024)) == 2)
        #expect(FieldControl.choiceIndex(.choice(nil), choices: choices, auto: false, defaultValue: .number(1024)) == 1)
        #expect(FieldControl.choiceIndex(.choice(9), choices: choices, auto: false, defaultValue: .number(1024)) == 1)
        #expect(FieldControl.choiceIndex(.text(""), choices: choices, auto: false, defaultValue: nil) == 0)
        // "Auto" listed (no default among the choices): unset stays Auto.
        #expect(FieldControl.choiceIndex(.choice(nil), choices: choices, auto: true, defaultValue: .number(768)) == -1)
        let field = FormField(key: "resolution", title: "Detail", help: nil, control: .choice(choices), required: false, defaultValue: .number(1024))
        let form = OperationForm(schema: [:])
        #expect(form.initialValue(field, setting: .number(768)) == .choice(1))
        #expect(form.initialValue(field, setting: nil) == .choice(1))
        let optional = FormField(key: "x", title: "X", help: nil, control: .choice(choices), required: false, defaultValue: nil)
        #expect(form.initialValue(optional, setting: .number(768)) == .choice(nil))
    }

    private func make3D(settings: [String: JSONValue]) throws -> CapabilitySession {
        let data = try Data(contentsOf: shippedOrRetiredFolder(Self.repo, "trellis2-4b").appendingPathComponent("ops/image_to_3d.input.json"))
        guard case .object(let schema) = try JSONDecoder().decode(JSONValue.self, from: data) else { throw WorkerFailure(code: "protocol", message: "schema") }
        let (s, _) = CapabilityWindows.make(
            .make3d, models: [CapabilityFixture.choice("trellis-fixture", resident: true)], selected: "trellis-fixture", client: { nil }, inbox: scratch,
            output: scratch, scratch: scratch)
        s.previewOptions(operations: [CapabilityFixture.descriptor("image_to_3d", input: schema)], selected: "image_to_3d", values: [:], settings: settings)
        return s
    }

    // MARK: A9 / B9. A 16-bit copy removed to save space

    /// The removed 16-bit cell is neither greyed (not offered) nor plain (never got): it carries the download arrow, its
    /// hover says it was removed and what Get downloads, and the action and Get more models say so too.
    @Test func aRemovedSixteenBitCopyIsShownInItsCell() throws {
        let row = try #require(Fixtures.rows(.retention).first { $0.id == "qwen3.8-27b" })
        let p = RowPresentation(row)
        #expect(ModelHelp.removed(row) == ["16"])
        #expect(ModelHelp.unavailable(row)["16"] == nil)
        let sixteen = ModelHelp.precision(row, column: "16", presentation: p)
        #expect(sixteen.contains("\n" + ModelHelp.removedLine + "\n") && sixteen.hasSuffix("Get downloads it again (55.6 GB)"), "\(sixteen)")
        // The derived cell is int4 (int8 has no cell): once the 16-bit source is released and int4 is not on
        // this Mac, its Get downloads the source again.
        var derived = row; derived.states["int4"] = PrecisionState(state: .notInstalled); derived.downloadBytes["int4"] = 55_600_000_000
        let derivedHover = ModelHelp.precision(derived, column: "4", presentation: RowPresentation(derived))
        #expect(derivedHover.hasSuffix("Get downloads the deleted 16-bit copy again (55.6 GB)"), "\(derivedHover)")
        #expect(!ModelHelp.precision(row, column: "4", presentation: p).contains("removed"))
        let chosen = RowPresentation(row, previewPrecision: "bf16")
        #expect(ModelHelp.action(row, presentation: chosen).contains("again") && ModelHelp.action(row, presentation: chosen).contains("save space"))
        let derivedAction = RowPresentation(derived, previewPrecision: "int4")
        #expect(ModelHelp.action(derived, presentation: derivedAction).hasPrefix("Downloads the 16-bit weights again"))
        // Without the phase, nothing says removed.
        var plain = row; plain.states["bf16"] = PrecisionState(state: .notInstalled)
        #expect(ModelHelp.removed(plain).isEmpty && !ModelHelp.precision(plain, column: "16", presentation: RowPresentation(plain)).contains("Removed"))
        var plainDerived = plain; plainDerived.states["int4"] = PrecisionState(state: .notInstalled); plainDerived.downloadBytes["int4"] = 55_600_000_000
        #expect(ModelHelp.precision(plainDerived, column: "4", presentation: RowPresentation(plainDerived)).hasSuffix("Get downloads 55.6 GB"))
        // Once Get brings it back, the cell is plain again.
        var back = row; back.states["bf16"] = PrecisionState(state: .ready)
        #expect(ModelHelp.removed(back).isEmpty)
    }

    /// The control draws the mark in the removed cell only, and a warning's triangle keeps its slot.
    @Test func theRemovedMarkIsDrawnInItsCell() throws {
        let row = try #require(Fixtures.rows(.retention).first { $0.id == "qwen3.8-27b" })
        let p = RowPresentation(row)
        let view = TierControl(
            labels: PrecisionColumn.all.map { PrecisionColumn.label(row, $0) }, selected: PrecisionColumn.of(p.selected, formats: row.formats), enabled: true,
            unavailable: ModelHelp.unavailable(row), warnings: ["4": "int4 may lose quality"], removed: ModelHelp.removed(row).union(["4"]),
            help: { ModelHelp.precision(row, column: $0, presentation: p) }, onSelect: { _ in })
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(origin: .zero, size: NSSize(width: TierControl.width, height: TierControl.height))
        let window = NSWindow(
            contentRect: NSRect(x: -9000, y: -9000, width: TierControl.width, height: TierControl.height), styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.contentView = host
        defer { window.close() }
        host.layoutSubtreeIfNeeded()
        let control = try #require(Self.views(host, TierSegments.Control.self).first)
        #expect(control.removedSegments == [0] && control.warnedSegments == [1])
        #expect(control.isEnabled(forSegment: 0))
        #expect(control.toolTip(forSegment: 0)?.contains(ModelHelp.removedLine) == true)
        #expect(TierSegments.Control.removedMark(.white) != nil)
    }

    static func views<T: NSView>(_ root: NSView, _: T.Type) -> [T] { (root as? T).map { [$0] } ?? root.subviews.flatMap { views($0, T.self) } }

    /// Get more models names the removed copy in its size lines.
    @Test func getMoreModelsSaysTheCopyWasRemoved() throws {
        let row = try #require(Fixtures.rows(.retention).first { $0.id == "qwen3.8-27b" })
        var entry = CatalogEntry(row: row)
        entry.derivedDtypes = ["int8", "int4"]
        let sizes = ModelDetail.sizes(entry)
        #expect(sizes.first { $0.dtype == "bf16" }?.note == "deleted to save space, download again", "\(sizes)")
    }

    // MARK: B8. Models whose 8- and 4-bit tiers were withdrawn, and Smart Turn in Speech

    /// A saved int8 or int4 for a model that is now 16-bit only (TIPSv2, Harrier, Jina, RIFE; Smart Turn on the development Mac
    /// holds int4) comes back as the 16-bit tier: the row chooses it, the cells read, nothing fails. Leftover 8- or 4-bit
    /// files are what the service reports; this records what it does with them today.
    @Test func aWithdrawnTierFallsBackToSixteenBit() async throws {
        let ids = ["tipsv2-l14", "harrier-oss-v1-0.6b", "jina-reranker-v3.5", "rife-4.25", "smart-turn-v3", "kokoro-82m"]
        let root = scratch.appendingPathComponent("real")
        let resources = try Self.bundle(ids, at: root)
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        do {
            let (service, error) = await ServiceFactory.make(paths: paths, environment: [:], bundle: resources)
            #expect(error == nil, "\(error ?? "")")
            for id in ids { try await service.add(id: id) }
            await service.shutdown()
        }
        // The saved choices as an older Arsenal left them, with that tier's files still on disk.
        let config = paths.support.appendingPathComponent("config.json")
        var json = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: config)) as? [String: Any])
        var selections = json["selections"] as? [String: Any] ?? [:]
        let saved = ["tipsv2-l14": "int8", "harrier-oss-v1-0.6b": "int4", "jina-reranker-v3.5": "int8", "rife-4.25": "int4", "smart-turn-v3": "int4"]
        for (id, dtype) in saved {
            var s = selections[id] as? [String: Any] ?? [:]
            s["precision"] = dtype; s["explicitSelection"] = true
            selections[id] = s
            let manifest = try ModelManifest.load(resources.appendingPathComponent("models/\(id)"))
            let leftover = paths.support.appendingPathComponent("weights/snapshots/\(id)/\(manifest.weights.revision)/\(dtype)")
            try FileManager.default.createDirectory(at: leftover, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: leftover.appendingPathComponent("arsenal-snapshot.json"))
            try Data(count: 1024).write(to: leftover.appendingPathComponent("model.safetensors"))
        }
        json["selections"] = selections
        try JSONSerialization.data(withJSONObject: json).write(to: config)

        let (service, error) = await ServiceFactory.make(paths: paths, environment: [:], bundle: resources)
        #expect(error == nil, "\(error ?? "")")
        let app = AppDelegate(service: service, paths: paths)
        defer { app.table.stop() }
        await app.table.refresh()
        for (id, dtype) in saved {
            let row = try #require(app.table.rows.first { $0.id == id })
            let p = app.table.presentation(row)
            #expect(!row.precisions.contains(dtype) && PrecisionColumn.of(p.selected, formats: row.formats) == "16", "\(id): \(row.selectedPrecision) of \(row.precisions)")
            #expect(row.selectedPrecision == p.selected, "\(id): the service still names \(row.selectedPrecision)")
            #expect(p.status.tone != .error && p.action == .get, "\(id): \(p.status.text) \(p.action)")
            #expect(ModelHelp.unavailable(row)["16"] == nil && !ModelHelp.precision(row, column: "16", presentation: p).isEmpty)
            for column in ["4"] { #expect(ModelHelp.unavailable(row)[column]?.hasPrefix("Not offered.") == true, "\(id) \(column)") }
            // No 8-bit column at all: an old 8-bit choice falls back to the 16-bit cell.
            #expect(ModelHelp.unavailable(row)["8"] == nil, "\(id): an 8-bit cell")
            for column in TableFigures.columns {
                let figure = TableFigures.figure(row, precision: p.selected, column: column, host: "M5 Max")
                #expect(figure.value != nil || !figure.help.isEmpty, "\(id) \(column) is blank")
            }
            #expect(row.states[dtype] == nil, "\(id): the withdrawn \(dtype) shows as a state")
        }
        // Smart Turn sits in Speech (it detects a spoken turn's end), its figures read in that group's columns.
        let speech = try #require(app.table.groups.first { $0.rows.contains { $0.id == "smart-turn-v3" } })
        #expect(speech.kind.id == "voices" && speech.kind.title == "Speech")
        let turn = try #require(speech.rows.first { $0.id == "smart-turn-v3" })
        for column in TableFigures.columns {
            let figure = TableFigures.figure(turn, precision: app.table.presentation(turn).selected, column: column, host: "M5 Max")
            #expect(figure.value != nil || figure.help.hasSuffix(TableFigures.notMeasured), "\(column): \(figure)")
        }
        GoNoGoProbe.say("smart-turn row: kind \(turn.kind) precisions \(turn.precisions) selected \(turn.selectedPrecision) metrics \(turn.metrics.keys.sorted())")
        GoNoGoProbe.say(
            "smart-turn speed: \(TableFigures.figure(turn, precision: "fp16", column: .speed, host: "M5 Max")) header: \(TableStats.help(kind: speech.kind, column: "speed"))")
        for (id, dtype) in saved {
            let leftover = try ModelManifest.load(resources.appendingPathComponent("models/\(id)")).weights.revision
            let there = FileManager.default.fileExists(atPath: paths.support.appendingPathComponent("weights/snapshots/\(id)/\(leftover)/\(dtype)").path)
            GoNoGoProbe.say("\(id) leftover \(dtype) after start: \(there ? "kept" : "removed")")
        }
        await service.shutdown()
    }

    // MARK: Renders

    /// `ARSENAL_GONOGO_RENDERS=<dir>`: the changed surfaces at 2× in dark and light, rendered offscreen (no window is
    /// ever ordered in): the table with a removed 16-bit copy, the table over the real service with the withdrawn tiers
    /// and Smart Turn in Speech, and Make 3D with a saved detail outside its choices. Hovers go to gonogo-hovers.txt.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_GONOGO_RENDERS"] != nil))
    func rendersTheChangedSurfaces() async throws {
        let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ARSENAL_GONOGO_RENDERS"]), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var hovers: [String] = []
        defer { Appearance.shared.set(.dark) }

        let rows = Fixtures.rows(.retention)
        let retention = TableController(service: FixtureService(rows: rows, kinds: Fixtures.kinds))
        retention.apply(rows: rows, kinds: Fixtures.kinds)
        if let row = rows.first(where: { $0.id == "qwen3.8-27b" }) {
            let p = RowPresentation(row)
            for column in PrecisionColumn.all { hovers.append("retention \(row.name) cell \(column):\n" + ModelHelp.precisionTooltip(row, column: column, presentation: p)) }
            hovers.append("retention action (bf16 chosen): " + ModelHelp.action(row, presentation: RowPresentation(row, previewPrecision: "bf16")))
            hovers.append("retention action (int8 chosen): " + ModelHelp.action(row, presentation: RowPresentation(row, previewPrecision: "int8")))
        }

        let ids = ["tipsv2-l14", "harrier-oss-v1-0.6b", "jina-reranker-v3.5", "rife-4.25", "smart-turn-v3", "kokoro-82m", "pocket-tts"]
        let root = scratch.appendingPathComponent("render-real")
        let resources = try Self.bundle(ids, at: root)
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        let (service, error) = await ServiceFactory.make(paths: paths, environment: [:], bundle: resources)
        #expect(error == nil, "\(error ?? "")")
        for id in ids { try await service.add(id: id) }
        let app = AppDelegate(service: service, paths: paths)
        await app.table.refresh()
        for group in app.table.groups where group.rows.contains(where: { $0.id == "smart-turn-v3" }) {
            hovers.append("group \(group.kind.title): " + group.rows.map(\.name).joined(separator: ", "))
            hovers.append("  Speed heading: " + TableStats.help(kind: group.kind, column: "speed"))
            for row in group.rows {
                let p = app.table.presentation(row)
                for column in TableFigures.columns {
                    let f = TableFigures.figure(row, precision: p.selected, column: column, host: "M5 Max")
                    hovers.append("  \(row.name) \(column.title): \(f.value ?? "triangle") | \(f.help.replacingOccurrences(of: "\n", with: " / "))")
                }
            }
        }

        let session = try #require(try make3D(settings: ["resolution": .number(768), "texture_size": .number(3000)]) as? Make3DSession)
        defer { session.discard() }
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode)
            await Self.renderTable(retention, to: directory.appendingPathComponent("gonogo-table-retention-\(mode.rawValue).png"))
            await Self.renderTable(app.table, to: directory.appendingPathComponent("gonogo-table-real-\(mode.rawValue).png"))
            let window = ToolWindowController(title: "Make 3D", autosave: "", size: ToolLayout.size, minSize: ToolLayout.minSize, content: Make3DView(session: session))
            try await Self.capture(try #require(window.window?.contentView), to: directory.appendingPathComponent("gonogo-make3d-\(mode.rawValue).png"))
            window.close()
        }
        app.table.stop()
        await service.shutdown()
        try hovers.joined(separator: "\n\n").write(to: directory.appendingPathComponent("gonogo-hovers.txt"), atomically: true, encoding: .utf8)
    }

    /// The table as the menu hosts it, on the menu's stand-in material.
    static func renderTable(_ controller: TableController, to url: URL) async {
        let table = NSHostingView(rootView: ModelsMenuView(controller: controller))
        table.frame = NSRect(x: 12, y: 10, width: ModelTable.width(controller), height: ModelTable.height(controller))
        let container = NSView(frame: NSRect(x: 0, y: 0, width: table.frame.width + 24, height: table.frame.height + 20))
        container.wantsLayer = true
        let dark = Appearance.shared.palette.dark
        MenuMock.applyMaterial(to: container.layer, dark: dark, picture: nil)
        container.layer?.cornerRadius = 10; container.layer?.masksToBounds = true
        container.addSubview(table)
        container.appearance = Appearance.shared.nsAppearance
        try? await capture(container, to: url) { MenuMock.standInMaterials(in: container, dark: dark, picture: nil) }
    }

    /// A view drawn into a bitmap from a window that is never ordered in (far off any screen).
    static func capture(_ view: NSView, to url: URL, prepare: (() -> Void)? = nil) async throws {
        var window: NSWindow? = view.window
        if window == nil {
            let w = NSWindow(contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: view.frame.size), styleMask: .borderless, backing: .buffered, defer: false)
            w.isReleasedWhenClosed = false; w.backgroundColor = .clear; w.appearance = Appearance.shared.nsAppearance; w.contentView = view
            window = w
        }
        defer { if view.window === window, window?.contentView === view, !(window is ToolWindow) { window?.close() } }
        try await Task.sleep(for: .milliseconds(400))
        view.layoutSubtreeIfNeeded(); prepare?()
        let rep = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: rep)
        try rep.representation(using: .png, properties: [:])?.write(to: url)
    }

    /// The app's own bundle for a real service: these model folders and the kinds under a dev-signed catalog.
    static func bundle(_ ids: [String], at root: URL) throws -> URL {
        let fm = FileManager.default
        let resources = root.appendingPathComponent("bundle/Contents/Resources")
        let kinds = resources.appendingPathComponent("kinds"), models = resources.appendingPathComponent("models")
        try fm.createDirectory(at: models, withIntermediateDirectories: true)
        try fm.copyItem(at: repo.appendingPathComponent("Resources/kinds"), to: kinds)
        func inventory(_ dir: URL) throws -> [CatalogFile] {
            try SafePath.files(in: dir).map {
                CatalogFile(path: $0, sha256: try FileDigest.sha256(dir.appendingPathComponent($0)), bytes: try FileDigest.bytes(dir.appendingPathComponent($0)))
            }
        }
        var entries: [CatalogModel] = []
        for id in ids {
            let folder = models.appendingPathComponent(id)
            try fm.copyItem(at: shippedOrRetiredFolder(repo, id), to: folder)
            entries.append(CatalogModel(manifest: try ModelManifest.load(folder, kindsDirectory: kinds), files: try inventory(folder)))
        }
        let data = try CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: entries, kindFiles: try inventory(kinds)).encoded()
        let key = Curve25519.Signing.PrivateKey()
        try fm.createDirectory(at: resources.appendingPathComponent("catalog"), withIntermediateDirectories: true)
        try data.write(to: resources.appendingPathComponent("catalog/catalog.json"))
        try key.signature(for: data).write(to: resources.appendingPathComponent("catalog/catalog.json.sig"))
        try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
        return resources
    }
}

/// Findings the tests print for the report (in the test log).
enum GoNoGoProbe {
    static func say(_ line: String) { print("gonogo: " + line) }
}
