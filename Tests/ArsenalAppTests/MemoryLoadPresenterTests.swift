// SPDX-License-Identifier: MIT
import AppKit
import ObjectiveC
import SwiftUI
import Testing
import ArsenalCore
import ArsenalAPI
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct MemoryLoadPresenterTests {
    let refusal = LoadMemoryRefusal(model: "flux2-klein-4b", name: "FLUX.2 Klein 4B", precision: "bf16", neededGB: 20.4, freeGB: 11.2, limitGB: 107.37, totalGB: 137.44)

    @Test func nativeAlertCopyAndCancelDefaultAreExact() {
        let alert = MemoryLoadPresenter.alert(refusal)
        #expect(alert.messageText == "Not enough free memory to load FLUX.2 Klein 4B")
        #expect(
            alert.informativeText
                == "Arsenal keeps the Mac at or under 100 GB of 128 GB in use, which keeps it out of swap. "
                + "This model needs about 19.0 GB at BF16; 10.4 GB fits under that limit now. Loading anyway lets macOS use swap, which slows the whole Mac.")
        #expect(alert.buttons.map(\.title) == ["Cancel", "Load Anyway"])
        #expect(alert.buttons.map(\.keyEquivalent) == ["\r", ""])
        #expect(alert.window.defaultButtonCell === alert.buttons[0].cell)
        #expect(alert.alertStyle == .warning)
        #expect(!alert.window.isVisible)
    }

    @Test func refusalMatchesTheChosenPrecisionsTablePeakEvenWhenLoadPeakIsLarger() {
        var row = ModelRow(id: "model", name: "Model", kind: "chat", precisions: ["bf16", "int4"])
        for (precision, peak, load) in [("bf16", 20.4, 25.0), ("int4", 3.123, 4.2)] {
            var stats = PrecisionStatSet(kind: .chat, dtype: precision, runner: row.runner)
            stats.set(.peakMemoryGB, StatValue(state: .quickCheck, value: peak))
            stats.set(.loadPeakMemoryGB, StatValue(state: .quickCheck, value: load))
            row.precisionStats[precision] = stats
            let check = LoadMemoryAssessment(row: row, precision: precision, forecast: .init(residentBytes: [precision: 90_000_000_000], basis: "forecast"), freeMB: 1000)
            #expect(check.neededGB == peak && check.basis == .peak)
            let figure = TableFigures.figure(row, precision: precision, column: .memory, host: nil)
            let refusal = LoadMemoryRefusal(model: row.id, name: row.name, precision: precision, neededGB: peak, freeGB: 1, limitGB: 107.37, totalGB: 137.44)
            #expect(figure.value == ModelTableFigure.memoryText(peak))
            let tail = "This model needs about \(figure.value ?? "missing") at \(precision.uppercased()); 0.93 GB fits under that limit now."
            #expect(refusal.explanation.hasSuffix(tail + " Loading anyway lets macOS use swap, which slows the whole Mac."))
        }
        // A declared forecast is consulted only after the table has no Peak memory figure.
        row.precisionStats.removeValue(forKey: "int4")
        let fallback = LoadMemoryAssessment(row: row, precision: "int4", forecast: .init(residentBytes: ["int4": 4_000_000_000], basis: "forecast"), freeMB: 1000)
        #expect(fallback.neededGB == 4 && fallback.basis == .forecast)
    }

    @Test func escapeActivatesTheNativeCancelButtonOffscreen() throws {
        let alert = MemoryLoadPresenter.alert(refusal)
        let target = CancelTarget()
        let cancel = alert.buttons[0]
        // Substitute the action receiver, never a modal session: exercise AppKit's actual key-equivalent routing.
        cancel.target = target; cancel.action = #selector(CancelTarget.cancel(_:))
        let escape = try #require(
            NSEvent.keyEvent(
                with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: alert.window.windowNumber,
                context: nil, characters: "\u{1b}", charactersIgnoringModifiers: "\u{1b}", isARepeat: false, keyCode: 53))
        #expect(alert.window.contentView?.performKeyEquivalent(with: escape) == true)
        #expect(target.calls == 1 && !alert.window.isVisible)
        #expect(alert.window.defaultButtonCell === cancel.cell)
    }

    @Test func menuLoadDismissesModelsAndMainThenActivatesBeforeTheResponder() async throws {
        let presentation = MemoryPresentationState()
        defer { presentation.restore() }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-memory-menu-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = MemoryNativeActionService(refusal: refusal)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root))
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh()
        let models = try #require(app.modelsMenu.modelItem().submenu)
        let method = try #require(class_getInstanceMethod(NSMenu.self, #selector(NSMenu.cancelTracking)))
        var order: [String] = []
        let replacement = imp_implementationWithBlock(
            { (menu: NSMenu) in
                if menu === models { order.append("dismiss Models") }
                if menu === app.menu { order.append("dismiss Main") }
            } as @convention(block) (NSMenu) -> Void)
        let original = method_setImplementation(method, replacement)
        defer { method_setImplementation(method, original); imp_removeBlock(replacement) }
        MemoryLoadPresenter.activate = { order.append("activate") }
        MemoryLoadPresenter.respond = { alert in
            order.append("respond")
            #expect(order == ["dismiss Models", "dismiss Main", "activate", "respond"])
            #expect(!alert.window.isVisible)
            return .alertFirstButtonReturn
        }
        app.table.perform(try #require(app.table.rows.first))
        try await waitUntil(timeout: 10) { order.last == "respond" }
        #expect(await service.attempts == [false])
        #expect(app.table.lastError == nil)
    }

    /// The window's line is one short sentence with the numbers; the alert's whole story is its hover; the action is the
    /// window's own Load Anyway (no Open Memory that only opens the menu).
    @Test func inlineMemoryRefusalIsOneLineWithLoadAnyway() throws {
        let error = ToolError.from(
            status: 409,
            body: try WireJSON.encode([
                "code": "insufficient_memory", "error": .string(refusal.failure.message), "details": .object(refusal.details)
            ]))
        let problem = OperationProblem.from(error, model: refusal.name)
        let alert = MemoryLoadPresenter.alert(refusal)
        #expect(problem.message == OperationProblem.memoryLine(refusal))
        #expect(problem.detail == alert.messageText + ". " + alert.informativeText)
        #expect(problem.action == .loadAnyway)
        #expect(!problem.message.contains("force: true"))
    }

    @Test(arguments: [false, true]) func presenterExercisesBothButtonsWithoutShowingAnAlert(_ anyway: Bool) async throws {
        let presentation = MemoryPresentationState()
        defer { presentation.restore() }
        var shown = 0
        MemoryLoadPresenter.respond = { alert in
            shown += 1
            #expect(!alert.window.isVisible)
            #expect(alert.buttons.count == 2)
            // Native alert result is injected: no modal session or on-screen window.
            return anyway ? .alertSecondButtonReturn : .alertFirstButtonReturn
        }
        var attempts: [Bool] = []
        do {
            let result = try await MemoryLoadPresenter.perform { force in
                attempts.append(force)
                if !force { throw refusal.failure }
                return "loaded"
            }
            #expect(anyway && result == "loaded")
        } catch is CancellationError { #expect(!anyway) }
        #expect(shown == 1)
        #expect(attempts == (anyway ? [false, true] : [false]))
        #expect(!LoadMemoryContext.force)
    }

    /// Core review 4: when unloading Arsenal's own idle models frees enough, the alert offers that besides Cancel and
    /// Load Anyway; choosing it unloads exactly those models and loads without swap. The refusal's words (window and API)
    /// name them, and a refusal they cannot fix offers only the two buttons.
    @Test func unloadingIdleModelsIsOfferedWhenItFreesEnough() async throws {
        let presentation = MemoryPresentationState()
        let unload = MemoryLoadPresenter.unloadModels, settle = MemoryLoadPresenter.settle
        defer { presentation.restore(); MemoryLoadPresenter.unloadModels = unload; MemoryLoadPresenter.settle = settle }
        let idle = LoadMemoryRefusal(
            model: "flux2-klein-4b", name: "FLUX.2 Klein 4B", precision: "bf16", neededGB: 20.4, freeGB: 11.2, limitGB: 107.37, totalGB: 137.44,
            idle: [.init(id: "kokoro-82m", name: "Kokoro 82M", gb: 0.8), .init(id: "qwen3-asr-1.7b", name: "Qwen3 ASR 1.7B", gb: 9.0)], usedMB: 90_000)
        #expect(LoadMemoryRefusal(details: idle.details) == idle)
        #expect(idle.idle.map(\.id) == ["qwen3-asr-1.7b", "kokoro-82m"] && idle.unloadingIdleFits)
        #expect(idle.explanation.contains("; 10.4 GB fits under that limit now. Unloading Arsenal's idle models, Qwen3 ASR 1.7B and Kokoro 82M (9.13 GB), frees enough."))
        #expect(idle.failure.message.hasSuffix(#"Or unload the idle models (POST /arsenal/v1/unload with {"id":"qwen3-asr-1.7b"}, {"id":"kokoro-82m"}) and retry."#))
        let short = LoadMemoryRefusal(
            model: "flux2-klein-4b", name: "FLUX.2 Klein 4B", precision: "bf16", neededGB: 20.4, freeGB: 11.2, idle: [.init(id: "kokoro-82m", name: "Kokoro 82M", gb: 0.8)])
        #expect(!short.unloadingIdleFits && !short.explanation.contains("Unloading") && !short.failure.message.contains("/unload"))
        #expect(MemoryLoadPresenter.alert(short).buttons.map(\.title) == ["Cancel", "Load Anyway"])
        let alert = MemoryLoadPresenter.alert(idle)
        #expect(alert.buttons.map(\.title) == ["Cancel", "Load Anyway", "Unload Idle Models and Load"])
        #expect(alert.buttons.map(\.keyEquivalent) == ["\r", "", ""] && alert.window.defaultButtonCell === alert.buttons[0].cell)

        MemoryLoadPresenter.settle = .zero
        MemoryLoadPresenter.respond = { _ in .alertThirdButtonReturn }
        var unloaded: [String] = []
        MemoryLoadPresenter.unloadModels = { unloaded += $0 }
        var attempts: [Bool] = []
        let result = try await MemoryLoadPresenter.perform { force in
            attempts.append(force)
            if attempts.count == 1 { throw idle.failure }
            return "loaded"
        }
        #expect(result == "loaded" && unloaded == ["qwen3-asr-1.7b", "kokoro-82m"] && attempts == [false, false])
    }

    @Test func toolErrorsReachTheSamePresenterAndUnrelatedFailuresDoNot() async throws {
        let presentation = MemoryPresentationState()
        defer { presentation.restore() }
        var shown = 0
        MemoryLoadPresenter.respond = { _ in
            shown += 1; return .alertSecondButtonReturn
        }
        let data = try WireJSON.encode(["error": .string(refusal.failure.message), "code": "insufficient_memory", "details": .object(refusal.details)])
        let error = ToolError.from(status: 409, body: data)
        #expect(error.details == refusal.details)
        let value = try await MemoryLoadPresenter.perform { force in
            if !force { throw error }; return 1
        }
        #expect(value == 1 && shown == 1)
        await #expect(throws: WorkerFailure.self) {
            try await MemoryLoadPresenter.perform { _ in throw WorkerFailure(code: "not_installed", message: "Get first") }
        }
        #expect(shown == 1)
        let admission = ToolCenter.Admission(operation: "generate", refusal: refusal.failure)
        #expect(admission.refusal == nil)
    }
    @Test(arguments: [false, true], [false, true])
    func toolTransportPresentsHTTPAndLateStreamRefusalsAndKeepsCancellation(_ stream: Bool, _ anyway: Bool) async throws {
        let presentation = MemoryPresentationState()
        defer { presentation.restore() }
        var shown = 0
        MemoryLoadPresenter.respond = { alert in
            #expect(!alert.window.isVisible); shown += 1
            return anyway ? .alertSecondButtonReturn : .alertFirstButtonReturn
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-memory-tool-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let server = ArsenalAPIServer(service: MemoryWarningTransportService(refusal: refusal), supportDirectory: root, port: 0)
        let port = try await server.start(); defer { server.stop() }
        let client = ArsenalClient(port: port)
        do {
            if stream {
                var messages = 0
                for try await _ in client.events("POST", "/arsenal/v1/models/model/ops/chat", json: [:], requireDone: false) { messages += 1 }
                #expect(anyway && messages == 1)
            } else {
                let (_, response) = try await client.send("POST", "/arsenal/v1/load", json: ["id": "model"])
                #expect(anyway && response.statusCode == 200)
            }
        } catch is CancellationError { #expect(!anyway) }
        #expect(shown == 1)
    }

    @Test(arguments: ["preview", "menu", "rollback"], [false, true])
    func nativeActionEntryPointsUseTheInjectedPresenter(_ route: String, _ anyway: Bool) async throws {
        let presentation = MemoryPresentationState()
        defer { presentation.restore() }
        var shown = 0
        MemoryLoadPresenter.respond = { alert in
            shown += 1; #expect(!alert.window.isVisible)
            return anyway ? .alertSecondButtonReturn : .alertFirstButtonReturn
        }
        let service = MemoryNativeActionService(refusal: refusal)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-native-memory-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        if route == "rollback" {
            let browser = BrowserController(service: service)
            await browser.refresh()
            browser.act("model", .rollback)
            try await waitUntil(timeout: 10) { await service.attempts.count == (anyway ? 2 : 1) && shown == 1 }
            #expect(browser.lastError == nil)
        } else {
            let app = AppDelegate(service: service, paths: ArsenalPaths(support: root))
            defer { app.runner.stop() }
            app.runner.writeClipboard = { _ in }
            app.runner.play = { _, _, _ in true }
            await app.table.refresh()
            let row = try #require(app.table.rows.first)
            if route == "preview" {
                app.table.toggleVoicePreview(row, "voice")
            } else {
                #expect(app.runner.run(LoadedAction(id: "menu", modelID: "model", title: "Answer", op: "chat")))
            }
            try await waitUntil(timeout: 10) { app.runner.running == nil && shown == 1 }
            #expect(app.runner.failure == nil)
            if !anyway { #expect(app.runner.lastResult == "Cancelled") }
        }
        #expect(await service.attempts == (anyway ? [false, true] : [false]))
        #expect(shown == 1 && !LoadMemoryContext.force)
    }

    @Test func unknownMemoryIsAQuietVisibleNativeLoadLine() async throws {
        let service = MemoryNativeActionService(refusal: refusal, unknown: true)
        let controller = TableController(service: service)
        await controller.refresh()
        let before = ModelTable.height(controller)
        let row = try #require(controller.rows.first)
        let presentation = MemoryPresentationState()
        defer { presentation.restore() }
        MemoryLoadPresenter.respond = { _ in
            Issue.record("Unknown memory must not open a modal"); return .alertFirstButtonReturn
        }
        controller.perform(row)
        try await waitUntil(timeout: 10) {
            let count = await service.attempts.count; return controller.memoryNotice != nil && count == 1
        }
        let line = try #require(ModelTable.footer(controller))
        #expect(line.text == "Model: " + LoadMemoryRequirement.unknownNote && !line.text.contains("GB"))
        #expect(controller.lastError == nil && ModelTable.height(controller) >= before)
        let host = NSHostingView(rootView: ModelTable(controller: controller))
        host.frame = NSRect(x: 0, y: 0, width: ModelTable.width(controller), height: ModelTable.height(controller))
        host.layoutSubtreeIfNeeded()
        let bitmap = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        #expect(bitmap.pixelsWide > 0 && bitmap.pixelsHigh >= Int(ModelTable.height(controller)))
        #expect(host.window?.isVisible != true)
    }

    @Test(arguments: [false, true]) func sharedToolRunPresentsOnceAndForcesOnlyItsRetry(_ anyway: Bool) async throws {
        let presentation = MemoryPresentationState()
        defer { presentation.restore() }
        var shown = 0
        MemoryLoadPresenter.respond = { alert in
            shown += 1; #expect(!alert.window.isVisible); return anyway ? .alertSecondButtonReturn : .alertFirstButtonReturn
        }
        let service = MemoryNativeActionService(refusal: refusal)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-kit-memory-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
        let port = try await server.start(); defer { server.stop() }
        let client = ArsenalClient(port: port)
        let session = ToolSession(
            models: [ModelChoice(model: ToolModel(id: "model", name: "Model", kind: "chat", dtype: "bf16", mode: "fast"))],
            selected: "model", client: { client }, inbox: root.appendingPathComponent("inbox"), outputDirectory: root, scratch: root.appendingPathComponent("scratch"))
        defer { session.discard() }
        session.operations = try await client.operations("model")
        session.operation = "chat"
        session.form = ToolForm(schema: try #require(session.descriptor).inputSchema)
        session.loading = false
        try #require(session.canRun)
        session.start()
        try await waitUntil(timeout: 10) { !session.running && shown == 1 }
        #expect(await service.attempts == (anyway ? [false, true] : [false]))
        #expect(session.run?.cancelled == !anyway && shown == 1)
        // Cancel says the run did not happen, with the alert's figures (and Load Anyway); Load Anyway leaves no line.
        #expect(session.problem == (anyway ? nil : OperationProblem.declinedRun(refusal)))
        if !anyway { #expect(session.problem?.message.hasPrefix("Not run: needs ") == true && session.problem?.message.contains(" of memory; ") == true) }
        // Load Anyway from the window's line forces only this new run.
        session.problem = OperationProblem.declined
        let count = await service.attempts.count
        session.performAction(.loadAnyway)
        try await waitUntil(timeout: 10) {
            let current = await service.attempts.count; return !session.running && current == count + 1
        }
        #expect(await service.attempts.last == true)
        #expect(shown == 1 && !LoadMemoryContext.force)
    }

}

/// runRefusal intentionally permits the preflight, reproducing RAM changing before the actual load.
private struct MemoryWarningTransportService: ModelsService {
    let refusal: LoadMemoryRefusal
    func rows() async -> [ModelRow] { [ModelRow(id: "model", name: "Model", kind: "chat", states: ["bf16": .init(state: .ready)])] }
    func kinds() async -> [KindSpec] { KindSpec.defaults }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws { if !LoadMemoryContext.force { throw refusal.failure } }
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        if !LoadMemoryContext.force { throw refusal.failure }; return ["text": "Hello"]
    }
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
    func operations(id: String) async -> [ModelOperation] {
        [ModelOperation(name: "chat", title: "Chat", inputSchema: ["type": "object"], output: "text-stream", stream: true, schemaVersion: 3)]
    }
}

private actor MemoryNativeActionService: ModelsService {
    let refusal: LoadMemoryRefusal
    let unknown: Bool
    var attempts: [Bool] = []
    init(refusal: LoadMemoryRefusal, unknown: Bool = false) { self.refusal = refusal; self.unknown = unknown }
    func rows() async -> [ModelRow] { [ModelRow(id: "model", name: "Model", kind: "voices", states: ["bf16": .init(state: .ready)])] }
    func catalog(query: CatalogQuery) async -> [CatalogEntry] { await rows().map { CatalogEntry(row: $0, added: true) } }
    func kinds() async -> [KindSpec] { KindSpec.defaults }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func loadMemoryNotice(id: String, precision: String?) async -> String? { unknown ? LoadMemoryRequirement.unknownNote : nil }
    func load(id: String, precision: String?, fast: Bool?) async throws { try record() }
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func rollbackRevision(id: String) async throws { try record() }
    private func record() throws { attempts.append(LoadMemoryContext.force); if !unknown && !LoadMemoryContext.force { throw refusal.failure } }
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject { try record(); return ["text": "Hello"] }
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
    func operations(id: String) async -> [ModelOperation] {
        [ModelOperation(name: "chat", title: "Chat", inputSchema: ["type": "object", "properties": [:]], output: "json", stream: false, schemaVersion: 3)]
    }
}

@MainActor private struct MemoryPresentationState {
    let respond = MemoryLoadPresenter.respond
    let activate = MemoryLoadPresenter.activate
    let dismiss = MemoryLoadPresenter.dismissMenus
    init() { MemoryLoadPresenter.activate = {} }
    func restore() {
        MemoryLoadPresenter.respond = respond; MemoryLoadPresenter.activate = activate; MemoryLoadPresenter.dismissMenus = dismiss
    }
}
@MainActor private final class CancelTarget: NSObject {
    var calls = 0
    @objc func cancel(_ sender: Any?) { calls += 1 }
}
