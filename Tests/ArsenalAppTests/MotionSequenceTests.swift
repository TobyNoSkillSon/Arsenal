// SPDX-License-Identifier: MIT
import AppKit
import Combine
import ArsenalCore
import Testing
@testable import Arsenal

#if DEBUG
    @MainActor @Suite(.menuState, .serialized) struct MotionSequenceTests {
        @Test(arguments: [CGFloat(0), CGFloat(400)])
        func tableTransitions(lostWidth: CGFloat) async throws {
            var row = ModelRow(id: "motion", name: "Motion", kind: "chat")
            let neighbour = ModelRow(id: "neighbour", name: "Neighbour", kind: "chat", states: ["bf16": .init(state: .ready)])
            let table = TableController(service: FixtureService(rows: []))
            defer { table.stop() }
            table.apply(rows: [row, neighbour], kinds: Fixtures.kinds, catalog: CatalogStatus(verified: true))
            let menu = ModelsMenu(controller: table)
            let submenu = try #require(menu.modelItem().submenu)
            let host = try #require(submenu.items.first?.view as? MenuTableHostingView)
            let harness = MotionFrames(host: host, lostWidth: lostWidth)
            defer { harness.close() }
            try await harness.expectFirstFrame("table-\(Int(lostWidth))")
            let suffix = "-\(Int(lostWidth))"
            try await harness.transition("submenu-open" + suffix) { menu.menuWillOpen(submenu) }
            try await harness.expectIdle("opened menu")
            var refreshes = 0
            menu.refreshModels = { refreshes += 1 }
            let refresh = try #require(MenuTests.find(RefreshControlView.self, in: host))
            let neighbourAction = try #require(allViews(host).compactMap { $0 as? RowActionView }.first { $0.spec?.title == "Load" })
            let neighbourGear = try #require(allViews(host).compactMap { $0 as? GearView }.last)
            harness.fixedControls = [refresh, neighbourAction, neighbourGear]
            let event = try #require(
                NSEvent.mouseEvent(
                    with: .mouseMoved, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0, context: nil,
                    eventNumber: 0, clickCount: 0, pressure: 0))
            try await harness.transition("refresh-hover" + suffix) { refresh.mouseEntered(with: event) }
            try await harness.transition("refresh-press" + suffix) { #expect(refresh.accessibilityPerformPress()) }
            #expect(refreshes == 1)
            try await harness.transition("refresh-exit" + suffix) { refresh.mouseExited(with: event) }
            try await harness.transition("refresh-disable" + suffix) { menu.updateRefresh(enabled: false, notice: nil) }
            #expect(!refresh.accessibilityPerformPress() && refreshes == 1)
            try await harness.transition("refresh-enable" + suffix) { menu.updateRefresh(enabled: true, notice: nil) }
            let action = try #require(allViews(host).compactMap { $0 as? RowActionView }.first { $0.spec?.title == "Get" })
            try await harness.transition("row-hover" + suffix) { action.mouseEntered(with: event) }
            // Invoke the same native control action, with a test sink so a press never starts a model or a download.
            var presses = 0
            if var spec = action.spec {
                spec = RowAction(title: spec.title, enabled: true, deletable: spec.deletable, help: spec.help, onPerform: { presses += 1 }, onDelete: {})
                action.spec = spec
            }
            try await harness.transition("row-press" + suffix) { #expect(action.accessibilityPerformPress()) }
            #expect(presses == 1)
            try await harness.transition("row-load-press" + suffix) { self.pressWithSink(neighbourAction) }
            try await harness.transition("row-delete-press" + suffix) { try self.pressGlyphWithSink(neighbourAction, cancel: false) }
            try await harness.transition("row-exit" + suffix) { action.mouseExited(with: event) }
            for state in [ModelState.downloading, .loading, .loaded] {
                try await harness.transition("row-\(state.rawValue)" + suffix) {
                    row.states["bf16"] = .init(state: state, progress: state == .downloading ? 0.35 : nil)
                    table.apply(rows: [row, neighbour], kinds: table.kinds, catalog: table.catalog)
                }
                let current = try #require(allViews(host).compactMap { $0 as? RowActionView }.first)
                try await harness.transition("row-\(state.rawValue)-hover" + suffix) { current.mouseEntered(with: event) }
                if state == .loaded {
                    #expect(current.spec?.title == "Unload")
                    try await harness.transition("row-unload-press" + suffix) { self.pressWithSink(current) }
                } else {
                    if state == .downloading {
                        try await harness.transition("row-cancel-press" + suffix) { try self.pressGlyphWithSink(current, cancel: true) }
                    }
                    try await harness.transition("row-\(state.rawValue)-press" + suffix) { #expect(!current.accessibilityPerformPress(), "Busy row refuses a press") }
                }
                try await harness.transition("row-\(state.rawValue)-exit" + suffix) { current.mouseExited(with: event) }
            }
            try await harness.transition("row-pending" + suffix) {
                row.needsReload = true; table.apply(rows: [row, neighbour], kinds: table.kinds, catalog: table.catalog)
            }
            #expect(table.presentation(row).pending)
            let reload = try #require(allViews(host).compactMap { $0 as? RowActionView }.first { $0.spec?.title == "Reload" })
            try await harness.transition("row-reload-press" + suffix) { self.pressWithSink(reload) }
            try await harness.transition("footer-error" + suffix) { table.lastError = "Motion fixture error" }
            try await harness.transition("footer-clear" + suffix) { table.lastError = nil }
            let added = ModelRow(id: "added", name: "Added", kind: "chat")
            try await harness.transition("row-arrives" + suffix) { table.apply(rows: [row, neighbour, added], kinds: table.kinds, catalog: table.catalog) }
            var publishes = 0
            let observation = table.objectWillChange.sink { publishes += 1 }
            defer { observation.cancel() }
            try await harness.transition("unchanged-refresh" + suffix) { table.apply(rows: table.rows, kinds: table.kinds, catalog: table.catalog) }
            #expect(publishes == 0, "Identical snapshots must not invalidate the table")
            try await harness.expectIdle("table settled")
        }

        private func pressWithSink(_ view: RowActionView) {
            guard let spec = view.spec else { Issue.record("Missing real row action"); return }
            var count = 0
            view.spec = RowAction(
                title: spec.title, busyText: spec.busyText, enabled: spec.enabled, deletable: spec.deletable,
                hot: spec.hot, prominent: spec.prominent, hovered: spec.hovered, help: spec.help, cancellable: spec.cancellable,
                onPerform: { count += 1 }, onDelete: {}, onCancel: {})
            #expect(view.accessibilityPerformPress())
            #expect(count == 1)
        }

        /// Real in-process hit testing at the native trash/cancel glyph, with no downloads or disk deletion.
        private func pressGlyphWithSink(_ view: RowActionView, cancel: Bool) throws {
            let original = try #require(view.spec)
            #expect(cancel ? original.cancellable : original.deletable)
            var performed = 0, deleted = 0, cancelled = 0
            view.spec = RowAction(
                title: original.title, busyText: original.busyText, enabled: original.enabled, deletable: original.deletable,
                hot: original.hot, prominent: original.prominent, hovered: original.hovered, help: original.help, cancellable: original.cancellable,
                onPerform: { performed += 1 }, onDelete: { deleted += 1 }, onCancel: { cancelled += 1 })
            defer { view.spec = original }
            let window = try #require(view.window)
            let point = view.convert(NSPoint(x: view.trashRect.midX, y: view.trashRect.midY), to: nil)
            let event = try #require(
                NSEvent.mouseEvent(
                    with: .leftMouseDown, location: point, modifierFlags: [], timestamp: 0, windowNumber: window.windowNumber,
                    context: nil, eventNumber: 0, clickCount: 1, pressure: 1))
            let frame = try #require(window.contentView?.superview)
            let hit = try #require(frame.hitTest(frame.convert(point, from: nil)))
            #expect(hit === view)
            hit.mouseDown(with: event)
            #expect(performed == 0 && deleted == (cancel ? 0 : 1) && cancelled == (cancel ? 1 : 0))
        }

        @Test func rowsArrivingBeforeAttachmentHaveTheCorrectFirstFrame() async throws {
            let table = TableController(service: FixtureService(rows: []))
            table.apply(rows: [], kinds: Fixtures.kinds, catalog: CatalogStatus(verified: true))
            let menu = ModelsMenu(controller: table)
            let host = try #require(menu.modelItem().submenu?.items.first?.view as? MenuTableHostingView)
            let row = ModelRow(id: "first-row", name: "First row", kind: "chat")
            table.apply(rows: [row], kinds: table.kinds, catalog: table.catalog)
            menu.resize()
            let harness = MotionFrames(host: host)
            defer { table.stop(); harness.close() }
            try await harness.expectFirstFrame("rows-before-attachment")
            #expect(host.naturalSize == NSSize(width: ModelTable.width(table), height: ModelTable.height(table)))
            let gear = try #require(MenuTests.find(GearView.self, in: host))
            let point = gear.convert(NSPoint(x: gear.bounds.midX, y: gear.bounds.midY), to: host)
            let hit = host.hitTest(host.convert(point, to: host.superview))
            #expect(hit === gear || hit?.isDescendant(of: gear) == true)
        }

        @Test func unchangedEntryPointsDoNotPublishOrLayout() async throws {
            let row = ModelRow(id: "entry", name: "Entry", kind: "chat")
            let table = TableController(service: FixtureService(rows: []))
            table.apply(rows: [row], kinds: Fixtures.kinds, catalog: CatalogStatus(verified: true))
            let menu = ModelsMenu(controller: table)
            let host = try #require(menu.modelItem().submenu?.items.first?.view as? MenuTableHostingView)
            let harness = MotionFrames(host: host)
            defer { table.stop(); harness.close() }
            try await harness.expectFirstFrame("unchanged-entry-points")
            var publishes = 0
            let observation = table.objectWillChange.sink { publishes += 1 }
            defer { observation.cancel() }
            let layouts = host.layoutPasses
            try await harness.transition("unchanged-refresh-availability") { menu.updateRefresh(enabled: true, notice: nil) }
            #expect(publishes == 0)
            #expect(host.layoutPasses == layouts)
            publishes = 0
            try await harness.transition("menu-closed-with-nothing-open") { table.discardPreviews() }
            #expect(publishes == 0)
            try await harness.expectIdle("unchanged entry points")
        }

        /// The gear opens the model's settings card beside it: the table's size and allocation stay exactly as they were,
        /// every pixel outside the open row and the card (with its shadow) stays, the gear and the row's action keep their
        /// frames and identity, nothing redraws afterwards, and the gear pressed again brings back the table's exact first
        /// pixels.
        @Test func theGearOpensTheSettingsCardWithoutMovingTheTable() async throws {
            let (table, row) = try Self.qwenSettingsTable()
            defer { table.stop() }
            let menu = ModelsMenu(controller: table)
            let host = try #require(menu.modelItem().submenu?.items.first?.view as? MenuTableHostingView)
            let harness = MotionFrames(host: host)
            defer { harness.close(); table.settingsCard.close() }
            try await harness.expectFirstFrame("settings")
            let closed = try harness.capture()
            let size = host.naturalSize, allocations = harness.sizes.count
            let gear = try #require(table.settingsCard.gear(row.id))
            let action = try #require(MenuTests.find(RowActionView.self, in: host))
            harness.fixedControls = [gear, action]
            let band = gear.convert(gear.bounds, to: host)
            func outside(_ frame: MotionFrames<ModelsMenuView>.Frame, card: CGRect) -> [Data] {
                // Above the row and the card, and below both.
                let top = min(band.minY, card.minY - 16), bottom = max(band.maxY, card.maxY + 16)
                return [
                    frame.region(NSRect(x: 0, y: 0, width: closed.size.width, height: max(0, top))),
                    frame.region(NSRect(x: 0, y: bottom, width: closed.size.width, height: max(0, closed.size.height - bottom)))
                ]
            }
            try await harness.transition("gear-open") { #expect(gear.accessibilityPerformPress()) }
            #expect(table.settingsCard.openID == row.id && gear.active)
            let card = try #require(table.settingsCard.frame(table))
            #expect(try outside(harness.capture(), card: card) == outside(closed, card: card) && host.naturalSize == size && harness.sizes.count == allocations)
            try await harness.expectIdle("settings card open")
            try await harness.transition("gear-again") { #expect(gear.accessibilityPerformPress()) }
            #expect(table.settingsCard.openID == nil && !gear.active)
            #expect(try harness.capture().bytes == closed.bytes && host.naturalSize == size && harness.sizes.count == allocations)
            try await harness.expectIdle("settings settled")
        }

        /// The card's own frames: its switch flips in one step; the table keeps its size; Restore defaults keeps its frame
        /// and identity; the name line and every row but the switched one keep their pixels; nothing redraws at rest.
        @Test func theSettingsCardChangesInPlace() async throws {
            let (table, row) = try await Self.qwenServedTable()
            defer { table.stop() }
            let menu = ModelsMenu(controller: table)
            let host = try #require(menu.modelItem().submenu?.items.first?.view as? MenuTableHostingView)
            table.openSettings(row)
            let harness = MotionFrames(host: host)
            defer { harness.close(); table.settingsCard.close() }
            try await harness.expectFirstFrame("settings-card")
            let base = host.naturalSize
            let card = try #require(table.settingsCard.frame(table))
            // The card's name line.
            harness.fixedRegions = [NSRect(x: card.minX, y: card.minY, width: card.width, height: SettingsCard.headerHeight)]
            let keep = try #require(SettingsCardHost.views(MenuSwitch.self, host).first { $0.exposed?.accessibilityLabel() == "Keep loaded" })
            try await harness.transition("card-switch") {
                // What AppKit's click does once its tracking ends: the switch flips and sends its action.
                keep.state = keep.state == .on ? .off : .on
                #expect(keep.sendAction(keep.action, to: keep.target))
            }
            #expect(table.settings[row.id]?.first { $0.id == TableController.keepKey }?.value == .bool(true))
            #expect(host.naturalSize == base && table.settingsCard.frame(table) == card)
            try await harness.expectIdle("card after a switch")
        }

        /// Qwen's real folder panel served by a fixture service, so a setting's write goes through and returns.
        static func qwenServedTable() async throws -> (TableController, ModelRow) {
            let folder = SettingsRenderDelegate.repo.appendingPathComponent("models/qwen3.8-27b")
            let source = try SettingsRenderDelegate.folder(folder)
            let metadata: JSONObject = [
                "setting": .array(source.settings.map { .object(SettingsRenderDelegate.declarationJSON($0)) }),
                "settings_toml": .string(try String(contentsOf: folder.appendingPathComponent(SettingsPanel.fileName), encoding: .utf8))
            ]
            var row = ModelRow(id: "settings-motion", name: source.name, kind: "chat", states: ["bf16": .init(state: .ready)])
            row.tools = source.tools
            // With the owner's other models beside it, so the table is as tall as a real one and the card fits without scrolling.
            let table = TableController(service: FixtureService(rows: [row] + Fixtures.rows(.loaded), metadata: [row.id: metadata]))
            await table.refresh()
            table.apply(rows: table.rows, kinds: table.kinds, catalog: CatalogStatus(verified: true))
            return (table, try #require(table.rows.first { $0.id == row.id }))
        }

        static func qwenSettingsTable() throws -> (TableController, ModelRow) {
            let folder = SettingsRenderDelegate.repo.appendingPathComponent("models/qwen3.8-27b")
            let source = try SettingsRenderDelegate.folder(folder)
            let row = ModelRow(id: "settings-motion", name: source.name, kind: "chat", states: ["bf16": .init(state: .ready)])
            let table = TableController(service: FixtureService(rows: []))
            table.apply(rows: [row] + Fixtures.rows(.loaded), kinds: Fixtures.kinds, catalog: CatalogStatus(verified: true))
            table.panels[row.id] = source.panel
            table.settings[row.id] = TableController.withKeep(
                source.settings.map {
                    SettingState(declaration: SettingsRenderDelegate.declarationJSON($0), value: $0.defaultValue.map(SettingsPanel.json) ?? .null)
                }, row)
            return (table, row)
        }

        @Test func oneServiceEventReadsOnceAndRepeatedReadFeedbackDoesNotPublish() async throws {
            let service = RefreshReadService()
            await service.setVerifiedCatalog()
            let table = TableController(service: service)
            table.start()
            defer { table.stop() }
            await service.waitForRowsRead(1)
            await table.waitUntilSettled()
            let menu = ModelsMenu(controller: table)
            let host = try #require(menu.modelItem().submenu?.items.first?.view as? MenuTableHostingView)
            let harness = MotionFrames(host: host)
            defer { harness.close() }
            try await harness.expectFirstFrame("service-event")
            let reads = await service.counts
            await service.setRowsError("Motion read error")
            var publishes = 0
            let observation = table.objectWillChange.sink { publishes += 1 }
            defer { observation.cancel() }
            try await harness.transition("service-error-feedback") { await service.stateChanged() }
            await table.waitUntilSettled()
            #expect(await service.counts[.rows] == reads[.rows, default: 0] + 1)
            #expect(publishes == 1, "One changed footer; read feedback cannot trigger a refresh cascade")
            publishes = 0
            let count = await service.counts
            try await harness.transition("service-repeated-error") { await service.stateChanged() }
            await table.waitUntilSettled()
            #expect(await service.counts[.rows] == count[.rows, default: 0] + 1)
            #expect(publishes == 0, "An identical read error must not invalidate the table")
            let settledReads = await service.counts
            try await harness.expectIdle("service events settled")
            #expect(await service.counts == settledReads, "No polling while idle")
        }

        @Test func activityUpdates() async throws {
            let table = TableController(service: FixtureService(rows: []))
            let menu = ActivityMenu(controller: table)
            let host = try #require(menu.item().submenu?.items.first?.view as? ActivityHostingView)
            let harness = MotionFrames(host: host)
            defer { menu.activity.hide(); harness.close(); table.stop() }
            harness.fixedRegions = [NSRect(x: 14, y: 13, width: 65, height: 15), NSRect(x: 194, y: 13, width: 65, height: 15)]
            harness.animatedMeterRegions = [NSRect(x: 74, y: 40, width: 96, height: 6), NSRect(x: 254, y: 40, width: 96, height: 6)]
            try await harness.expectFirstFrame("activity")
            var feed = ActivityFeed(
                models: [.init(id: "image", kind: "images", state: "running", requests: [], queued: 0, batch: nil, memoryMB: 4000)],
                machine: .init(cpuBusyPct: 10, gpuBusyPct: 20, memoryUsedMB: 32000, memoryTotalMB: 128000, arsenalMB: 200))
            let initialResizes = harness.sizes.count
            try await harness.transition("activity-request-start", animated: true) {
                feed.models[0].requests = [.init(id: "request", requester: "Fixture", op: "generate", phase: "running", step: 1, total: 10)]
                menu.activity.preview(feed); menu.resize()
            }
            #expect(harness.sizes.count == initialResizes + 1, "Activity's explicit growth is counted")
            let progressResizes = harness.sizes.count
            try await harness.transition("activity-progress", animated: true) {
                feed.models[0].requests[0].step = 6; feed.machine.gpuBusyPct = 65
                menu.activity.preview(feed)
            }
            #expect(harness.sizes.count == progressResizes, "Progress changes no allocation")
            try await harness.expectIdle("Activity between updates")
            try await harness.transition("activity-request-end") {
                feed.models[0].requests = []; menu.activity.preview(feed); menu.resize()
            }
            #expect(harness.sizes.count == progressResizes + 1, "Activity's explicit shrink is counted")
            try await harness.expectIdle("Activity idle")
        }

        /// A one-second tick with a busy decision model and its queue draining (refresh "every second"; "how
        /// many are queued"): its counts, minute figures and graph change in place in one step, the allocation stays,
        /// nothing draws between ticks; the drained queue reads Idle in the same place; only the empty minute shrinks the
        /// card, once.
        @Test func activitySecondTick() async throws {
            let table = TableController(service: FixtureService(rows: []))
            let menu = ActivityMenu(controller: table)
            let host = try #require(menu.item().submenu?.items.first?.view as? ActivityHostingView)
            let harness = MotionFrames(host: host)
            defer { menu.activity.hide(); harness.close(); table.stop() }
            harness.fixedRegions = [NSRect(x: 14, y: 13, width: 65, height: 15), NSRect(x: 194, y: 13, width: 65, height: 15)]
            harness.animatedMeterRegions = [NSRect(x: 74, y: 40, width: 96, height: 6), NSRect(x: 254, y: 40, width: 96, height: 6)]
            var busy = [Double](repeating: 0, count: 60)
            for i in 12..<60 { busy[i] = 1 }
            var feed = ActivityFeed(
                models: [
                    .init(
                        id: "decider", kind: "decisions", state: "prefill", requests: [], queued: 40, batch: nil, memoryMB: 1200, running: 1,
                        recent: .init(finished: 250, meanRunS: 0.192, meanWaitS: 7.4, busyS: 48, busy: busy))
                ],
                machine: .init(cpuBusyPct: 20, gpuBusyPct: 80, cpuTempC: 50, gpuTempC: 70, memoryUsedMB: 60000, memoryTotalMB: 128000, arsenalMB: 300))
            try await harness.transition("activity-decisions-arrive", animated: true) {
                menu.activity.preview(feed); menu.resize()
            }
            try await harness.expectIdle("Activity between seconds")
            let sizes = harness.sizes.count
            // The models' second alone: one forward switch of every pixel, no intermediate frame.
            try await harness.transition("activity-tick-models") {
                feed.models[0].queued = 35
                feed.models[0].recent?.finished = 255; feed.models[0].recent?.busy = Array(busy.dropFirst()) + [1]
                menu.activity.preview(feed)
            }
            // With the machine's reading: its meters switch in place with their numbers, everything at once, no eased frame.
            try await harness.transition("activity-tick-machine") {
                feed.machine.cpuBusyPct = 26; feed.machine.gpuBusyPct = 84
                feed.models[0].queued = 30; feed.models[0].recent?.finished = 260
                menu.activity.preview(feed)
            }
            #expect(harness.sizes.count == sizes, "A second's new figures change no allocation")
            try await harness.expectIdle("Activity after a tick")
            // The queue drains: the counts line reads Idle in its place; nothing moves or resizes.
            try await harness.transition("activity-queue-drained") {
                feed.models[0].running = 0; feed.models[0].queued = 0; feed.models[0].state = "idle"
                menu.activity.preview(feed); menu.resize()
            }
            #expect(harness.sizes.count == sizes, "The drained queue keeps the card's height")
            try await harness.expectIdle("Activity idle after the queue")
            // A minute later the minute is empty: the card shrinks once.
            try await harness.transition("activity-minute-empty") {
                feed.models[0].recent = nil
                menu.activity.preview(feed); menu.resize()
            }
            #expect(harness.sizes.count == sizes + 1, "The card's explicit shrink is counted")
            try await harness.expectIdle("Activity idle after the minute")
        }
    }

#endif
