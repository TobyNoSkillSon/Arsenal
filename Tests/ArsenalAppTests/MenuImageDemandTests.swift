// SPDX-License-Identifier: MIT
import AppKit
import CryptoKit
@testable import ArsenalCore
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct MenuImageDemandTests {
    /// The real table event loop must settle after a persistent descriptor failure, including hidden tools.
    @Test(arguments: [true, false])
    func failedPrimaryLookupDoesNotRefreshItself(shown: Bool) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-lookup-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = try await fixture(root, generic: true, unavailableSchema: true)
        let source = try #require(await service.source as? MenuSchemaFailureSource)
        try await service.setToolsShown(id: "fixture-images", on: shown)
        let initialReads = await source.failedReads
        let stream = await service.events()
        let events = Task {
            var count = 0; for await _ in stream { count += 1 }; return count
        }
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root.appendingPathComponent("support")))
        defer { app.table.stop(); app.tools.closeAll(); events.cancel() }
        var refreshes = 0, menuRefreshes = 0
        // AppDelegate's production subscription, without creating a status item or showing a menu.
        app.table.start {
            refreshes += 1
            await app.refreshMenuState(); menuRefreshes += 1
        }
        try await waitUntil { menuRefreshes > 0 && menuRefreshes == refreshes }
        // Subscription supplies one initial catalog event as well as the table's initial refresh.
        try await Task.sleep(for: .milliseconds(25))
        let settledRefreshes = refreshes, settledMenus = menuRefreshes
        #expect(settledRefreshes == 1 && settledMenus == 1)
        let row = try #require(app.table.rows.first)
        let answer = try #require(app.tools.admissions[row.id]?[.operation])
        #expect(answer.refusal?.code == "fixture_schema")
        #expect(answer.refusal?.message == MenuSchemaFailureSource.message)
        #expect(app.tools.blocked(row)?.message == MenuSchemaFailureSource.message)
        let item = app.menu.items.first { $0.representedObject as? [String] == ["operation", row.id] }
        #expect(item == nil, "A tool with no runnable schema and no menu repair is omitted")
        #expect(await service.supervisor.residentIDs().isEmpty)
        // A loop would keep reading even with every answer unchanged. Observe beyond multiple refresh turns.
        try await Task.sleep(for: .milliseconds(250))
        #expect(refreshes == settledRefreshes && menuRefreshes == settledMenus)
        #expect(await source.failedReads - initialReads == settledMenus)
        app.table.stop(); events.cancel()
        #expect(await events.value == 1, "Only the subscription’s initial catalog event is allowed")
        await service.shutdown()
    }

    @Test func statusReadFailureDoesNotRefreshItself() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-status-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = try await fixture(root)
        let local = root.appendingPathComponent("support/local-models")
        if FileManager.default.fileExists(atPath: local.path) { try FileManager.default.removeItem(at: local) }
        try Data("Not a directory".utf8).write(to: local)
        let stream = await service.events()
        let events = Task {
            var count = 0; for await _ in stream { count += 1 }; return count
        }
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root.appendingPathComponent("support")))
        defer { app.table.stop(); app.tools.closeAll(); events.cancel() }
        var refreshes = 0, menuRefreshes = 0
        app.table.start {
            refreshes += 1
            await app.refreshMenuState(); menuRefreshes += 1
        }
        try await waitUntil { menuRefreshes > 0 && menuRefreshes == refreshes }
        // Subscription supplies one initial catalog event as well as the table's initial refresh.
        try await Task.sleep(for: .milliseconds(25))
        let settledRefreshes = refreshes, settledMenus = menuRefreshes
        #expect(settledRefreshes == 1 && settledMenus == 1)
        #expect(await service.catalogStatus().error?.hasPrefix("Local models:") == true)
        try await Task.sleep(for: .milliseconds(250))
        #expect(refreshes == settledRefreshes && menuRefreshes == settledMenus)
        // A later intentional refresh retries the read and clears the retained reason after repair.
        try FileManager.default.removeItem(at: local)
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        await app.table.refresh(); await app.refreshMenuState()
        #expect(await service.catalogStatus().error == nil)
        app.table.stop(); events.cancel()
        #expect(await events.value == 1, "Only the subscription’s initial catalog event is allowed")
        await service.shutdown()
    }

    @Test func loadedLocalVoiceDiscoveryFailureIsAnEventFreeRead() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-local-voice-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = try await fixture(root, voice: true)
        try await service.load(id: "fixture-images")
        let reply = try await service.voiceInventory(id: "fixture-images")
        #expect(!VoiceChoice.list(reply).isEmpty)
        let localRoot = root.appendingPathComponent("support/local-models")
        try FileManager.default.createDirectory(at: localRoot, withIntermediateDirectories: true)
        let local = localRoot.appendingPathComponent("fixture-images")
        try FileManager.default.copyItem(at: root.appendingPathComponent("Resources/models/fixture-images"), to: local)
        try await service.menuConvertToLocal(local)
        try FileManager.default.removeItem(at: localRoot)
        try Data("Not a directory".utf8).write(to: localRoot)
        let stream = await service.events()
        let events = Task {
            var count = 0; for await _ in stream { count += 1 }; return count
        }
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root.appendingPathComponent("support")))
        defer { app.table.stop(); app.tools.closeAll(); events.cancel() }
        var completed = 0
        app.table.start {
            await app.refreshMenuState(); completed += 1
        }
        try await waitUntil { completed == 1 && app.tools.errors["fixture-images"] != nil }
        let reason = try #require(app.tools.errors["fixture-images"])
        #expect(reason.hasPrefix("Voices unavailable: "))
        #expect(app.menu.items.contains { $0.title == reason && !$0.isEnabled })
        try await Task.sleep(for: .milliseconds(250))
        #expect(completed == 1 && app.tools.errors["fixture-images"] == reason)
        app.table.stop(); events.cancel()
        let eventCount = await events.value
        #expect(eventCount == 1, "No voice-read error event, only the initial catalog snapshot (saw \(eventCount))")
        await service.shutdown()
        #expect(await service.supervisor.residentIDs().isEmpty)
    }

    /// A real service and supervised stub executable: the menu opens the tool without loading;
    /// Generate uses HTTP and loads on demand. Repeat with the worker already resident.
    @Test func imagesLoadOnFirstRequestAndRunAgainWhileLoaded() async throws {
        _ = NSApplication.shared
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-demand-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = try await fixture(root)
        do {
            let app = AppDelegate(service: service, paths: ArsenalPaths(support: root.appendingPathComponent("support")))
            await app.api.start()
            defer { app.api.stop() }
            _ = try #require(app.api.port)
            app.tools.present = { _ in }
            app.tools.imageDirectory = root.appendingPathComponent("Pictures")
            defer { app.tools.closeAll(); app.table.stop() }
            await app.table.refresh(); await app.refreshMenuState()
            for loaded in [false, true] {
                let row = try #require(app.table.rows.first)
                #expect((row.residentDType != nil) == loaded)
                if loaded {
                    var pending = row
                    pending.needsCodeConsent = true; pending.selectedPrecision = "not-installed"
                    #expect(app.tools.blocked(pending, tool: .image) == nil, "The API keeps running the verified resident snapshot")
                }
                let index = try #require(app.menu.items.firstIndex { $0.representedObject as? String == AppDelegate.taskKey(.tool(.image)) })
                #expect(app.menu.items[index].isEnabled)
                app.menu.performActionForItem(at: index)
                let session = try #require(app.tools.image?.session)
                #expect(session.selected == row.id)
                try await waitUntil { !session.loading }
                #expect(await service.supervisor.residentIDs().isEmpty == !loaded)
                session.set("prompt", .text("A lighthouse")); session.set("seed", .text("42"))
                session.start()
                #expect(session.running && session.run?.finished == nil)
                try await waitUntil { !session.running }
                #expect(session.problem == nil, "\(session.problem?.message ?? "")")
                let result = try #require(session.results.last)
                #expect(result.seed == 42 && NSImage(contentsOf: result.url) != nil)
                #expect(session.results.count == (loaded ? 2 : 1))
                #expect(await service.rows().first?.residentDType == "bf16")
                #expect(await service.residents[row.id]?.residency == .onDemand)
                await app.table.refresh(); await app.refreshMenuState()
            }
            app.tools.closeAll()
            await service.shutdown()
            #expect(await service.supervisor.residentIDs().isEmpty)
        } catch {
            await service.shutdown()
            throw error
        }
    }

    /// Refusals show in the menu before a window opens, as the entry's note and hover, and the window's model choice
    /// carries the same problem with its one fix; the entry still opens the window. The API refuses the same inputs.
    @Test(arguments: ["missing", "consent", "selected-missing", "unsupported"])
    func refusedImagesStayDisabledWithTheReason(reason: String) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-refusal-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "images", states: ["bf16": .init(state: .ready)])
        row.tools = ["image"]
        switch reason {
        // On the table without weights: its Get failed (a model never fetched is only in Get more models).
        case "missing": row.states = ["bf16": .init(state: .failed, error: "Get failed: the connection stalled.")]
        case "consent": row.local = true; row.needsCodeConsent = true
        case "selected-missing": row.precisions = ["bf16", "int4"]; row.selectedPrecision = "int4"
        default: row.formats = [.init(id: "bf16", bitsPerWeight: 16, source: .vendor, runners: [])]
        }
        let service = FixtureService(rows: [row], supportDirectory: root)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root))
        await app.table.refresh(); await app.refreshMenuState()
        defer { app.table.stop(); app.tools.closeAll() }
        let item = try #require(app.menu.items.first { $0.representedObject as? String == AppDelegate.taskKey(.tool(.image)) })
        let problem = try #require(app.tools.blocked(row, tool: .image))
        let current = try #require(app.table.rows.first)
        let ready = try #require(app.tools.readiness(current, task: .tool(.image)))
        #expect(item.title == "Make an Image\u{2026}" && item.isEnabled && item.action != nil && !item.isHidden)
        #expect(!item.title.contains("Load") && !item.title.contains("Get"))
        switch reason {
        case "consent":
            #expect(item.badge?.stringValue == "Approve" && ready.problem.action == .approve)
            #expect(item.toolTip == ready.hover)
        case "unsupported":
            // No fix exists: the note says so and the hover gives the reason, as the API words it.
            #expect(item.badge?.stringValue == "Unavailable" && ready.problem.action == nil)
            #expect(ready.problem.message == problem.message && item.toolTip == problem.message)
        case "missing":
            // On the table without weights is a failed Get: the note says so and the window tries again.
            #expect(item.badge?.stringValue == "Failed" && ready.problem.action == .download && ready.problem.actionTitle == "Try Again")
            #expect(item.toolTip == ready.hover && ready.hover?.contains("The download stopped") == true)
        default:
            // Not at the chosen precision: no note; the window offers Download.
            #expect(item.badge == nil && ready.problem.action == .download && ready.problem.actionTitle?.hasPrefix("Download") == true)
            #expect(item.toolTip == ready.hover && ready.hover?.contains("Not on this Mac") == true)
        }
        #expect(app.tools.models(app.table.rows, for: .tool(.image)).first?.unavailable == ready.problem)
        #expect(app.tools.image == nil)
        if reason == "consent" {
            let api = APIHost(service: service, support: root, port: 0)
            await api.start(); defer { api.stop() }
            let client = ArsenalClient(port: try #require(api.port))
            do {
                _ = try await client.send("POST", "/arsenal/v1/models/fixture/ops/generate", json: ["prompt": "A lighthouse", "steps": 1])
                Issue.record("A request cannot approve local code")
            } catch let error as ToolError { #expect(error.code == "code_consent") }
            #expect(await service.rows().first?.residentDType == nil)
        }
        // Use the production service for missing-installation refusal, not a permissive fake.
        if reason == "missing" {
            let real = try await fixture(root.appendingPathComponent("real"), installed: false)
            do {
                let api = APIHost(service: real, support: root.appendingPathComponent("real/support"), port: 0)
                await api.start(); defer { api.stop() }
                let client = ArsenalClient(port: try #require(api.port))
                do {
                    _ = try await client.send("POST", "/arsenal/v1/models/fixture-images/ops/generate", json: ["prompt": "A lighthouse", "steps": 1])
                    Issue.record("An uninstalled model must not load or download")
                } catch let error as ToolError {
                    #expect(error.status == 404 && error.code == "not_installed", "\(error.message)")
                }
                #expect(await real.supervisor.residentIDs().isEmpty)
                await real.shutdown()
            } catch { await real.shutdown(); throw error }
        }
    }

    /// The menu's service answer versus a real HTTP run, including revision recovery.
    @Test(arguments: [
        "cold", "resident", "recovery", "not-downloaded", "busy-get", "resident-get", "updating",
        "resident-updating"
    ])
    func admissionMatchesHTTPInEveryReadinessState(state: String) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-admission-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = try await fixture(root, installed: state != "not-downloaded")
        do {
            let id = "fixture-images"
            if state.hasPrefix("resident") || state == "recovery" { try await service.load(id: id, precision: nil, fast: nil) }
            if state == "recovery" { try await service.menuSaveRecovery(id) }
            await service.menuAdmissionState(id, state)
            let app = AppDelegate(service: service, paths: ArsenalPaths(support: root.appendingPathComponent("support")))
            await app.api.start()
            defer { app.api.stop(); app.table.stop(); app.tools.closeAll() }
            let before = await service.supervisor.residentIDs()
            await app.table.refresh(); await app.refreshMenuState()
            #expect(await service.supervisor.residentIDs() == before, "Admission must not load a worker")
            // The table holds only models with weights or a Get under way: one not downloaded has no row and no menu entry
            // (Get more models has it); its admission is still read from its catalog row and matches HTTP.
            let onTable = state != "not-downloaded"
            #expect((app.table.rows.first != nil) == onTable, "\(state)")
            let row = try #require(onTable ? app.table.rows.first : await service.catalog().first { $0.id == id }?.row)
            if state == "recovery" {
                #expect(row.residentDType == nil)
                #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("support/state/revisions/\(id).json").path))
            }
            let item = app.menu.items.first { $0.representedObject as? String == AppDelegate.taskKey(.tool(.image)) }
            #expect((item != nil) == onTable, "\(state): the entry exists exactly while a table model makes images")
            let answer = await service.runRefusal(id: id, op: "generate")
            let expected: String? =
                switch state {
                case "not-downloaded": "not_installed"
                case "busy-get": "busy"
                case "updating", "resident-updating": "updating"
                default: nil
                }
            #expect(answer?.code == expected)
            let recoverable = ["not_installed"].contains(answer?.code ?? "")
            // The entry opens its window in every state; what the menu says matches what HTTP answers.
            let ready = app.tools.readiness(row, task: .tool(.image))
            if let item {
                #expect(item.isEnabled && item.action != nil, "\(state)")
                #expect(item.badge?.stringValue == ready?.note, "\(state)")
            }
            #expect((ready == nil) == (answer == nil), "\(state): menu \(ready?.note ?? "ready"), HTTP \(answer?.code ?? "accepted")")
            if recoverable { #expect(ready?.problem.action == .download && item?.badge == nil) }
            // The menu's admission covers its entries, so only a table model's is compared with HTTP.
            if onTable { #expect(app.tools.blocked(row, tool: .image)?.message == answer?.message) }
            if let answer, ready?.note == "Unavailable" { #expect(item?.toolTip == answer.message && ready?.problem.message == answer.message) }
            let client = ArsenalClient(port: try #require(app.api.port))
            do {
                _ = try await client.send("POST", "/arsenal/v1/models/\(id)/ops/generate", json: ["prompt": "A lighthouse", "steps": 1])
                #expect(answer == nil, "HTTP accepted a menu refusal")
                #expect(await service.rows().first?.residentRunner == "native")
            } catch let error as ToolError {
                #expect(answer != nil, "HTTP refused an enabled tool: \(error.message)")
                #expect(error.code == answer?.code && error.message == answer?.message)
            }
            await service.menuAdmissionState(id, "clear")
            await service.shutdown()
            #expect(await service.supervisor.residentIDs().isEmpty)
        } catch { await service.menuAdmissionState("fixture-images", "clear"); await service.shutdown(); throw error }
    }

    @Test func operationAdmissionNamesAnUnsupportedOperationWithoutLoading() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-op-admission-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = try await fixture(root)
        do {
            let api = APIHost(service: service, support: root.appendingPathComponent("support"), port: 0)
            await api.start(); defer { api.stop() }
            let answer = try #require(await service.runRefusal(id: "fixture-images", op: "absent"))
            #expect(answer.code == "unsupported_op")
            let client = ArsenalClient(port: try #require(api.port))
            do {
                _ = try await client.send("POST", "/arsenal/v1/models/fixture-images/ops/absent", json: [:])
                Issue.record("Unsupported operation was admitted")
            } catch let error as ToolError { #expect(error.status == 404 && error.message.contains("does not declare operation")) }
            #expect(await service.supervisor.residentIDs().isEmpty)
            await service.shutdown()
        } catch { await service.shutdown(); throw error }
    }

    private func waitUntil(_ condition: @MainActor () async -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        while !(await condition()) {
            try #require(ContinuousClock.now < deadline, "Image request did not finish")
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    /// Signed, three-byte weight fixture; no network, model download or inference.
    @Test func admissionFixtureUsesItsOwnMemoryBudget() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-memory-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = try await fixture(root)
        #expect(await service.memory().availableMB == 38000) // Raw no-swap figure: 10 GB of room plus the 28 GB above the line.
        do {
            try await service.load(id: "fixture-images", precision: nil, fast: nil)
            #expect(await service.supervisor.residentIDs() == ["fixture-images"])
            await service.shutdown()
        } catch { await service.shutdown(); throw error }
    }

    private func fixture(_ root: URL, installed: Bool = true, generic: Bool = false, unavailableSchema: Bool = false, voice: Bool = false) async throws -> RealModelsService {
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        try paths.create()
        let folder = root.appendingPathComponent("Resources/models/fixture-images")
        let weights = Data([12, 34, 56]), hash = FileDigest.sha256(weights), revision = String(repeating: "a", count: 40)
        let toml = """
            schema = 1
            id = "fixture-images"
            name = "Fixture Images"
            kind = "\(voice ? "voices" : "images")"
            tools = ["\(voice ? "voice" : generic ? "operation" : "image")"]
            summary = "Stub-only menu fixture."
            publisher = "Arsenal"
            released = 2026
            tasks = ["fixture"]
            languages = ["en"]
            params = "0"
            [licence]
            spdx = "MIT"
            commercial = true
            url = "https://example.com/licence"
            [weights]
            repo = "test/fixture"
            revision = "\(revision)"
            dtype = "bf16"
            files = ["base.bin"]
            [weights.sha256]
            "base.bin" = "\(hash)"
            [weights.bytes]
            "base.bin" = 3
            [[runner]]
            name = "native"
            language = "swift"
            worker = "stub"
            protocol = 1
            chips = ["*"]
            min_macos = "26.0"
            fast_exact = false
            [selection]
            default = "native"
            [[op]]
            name = "\(voice ? "voices" : "generate")"
            title = "Inventory"
            input = "ops/generate.input.json"
            output = "\(voice ? "json" : "image")"
            stream = false
            [gate]
            recipe = "gate/gate.toml"
            [bench]
            recipe = "bench/bench.toml"
            """
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let files: [String: Data] = [
            "model.toml": Data(toml.utf8), "gate/gate.toml": Data("schema=1".utf8), "bench/bench.toml": Data("schema=1".utf8),
            "README.md": Data("Stub-only fixture; no real model.".utf8), "log/001-fixture.md": Data("Three-byte, supervised stub fixture.".utf8),
            "ops/generate.input.json": try Data(
                contentsOf: repo.appendingPathComponent(voice ? "Resources/kinds/voices/ops/voices.input.json" : "Resources/kinds/images/ops/generate.input.json"))
        ]
        for (path, bytes) in files {
            let file = folder.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: file)
        }
        let model = try ModelManifest.load(folder)
        let catalogModel = CatalogModel(manifest: model, files: files.map { CatalogFile(path: $0.key, sha256: FileDigest.sha256($0.value), bytes: Int64($0.value.count)) })
        let catalog = CatalogDocument(sequence: 1, sourceCommit: revision, models: [catalogModel])
        let key = Curve25519.Signing.PrivateKey(), data = try catalog.encoded()
        if installed {
            let file = root.appendingPathComponent("base.bin"); try weights.write(to: file)
            let store = BlobStore(support: paths.support)
            try await store.promote(file, sha256: hash, bytes: 3)
            let snapshot: JSONObject = [
                "model": "fixture-images", "revision": .string(revision), "precision": "bf16",
                "files": [["path": "base.bin", "sha256": .string(hash), "bytes": 3]]
            ]
            try await store.commitSnapshot(JSONDecoder().decode(SnapshotManifest.self, from: WireJSON.encode(snapshot)))
        }
        var config = ArsenalConfiguration(); config.catalog = ["source": "bundle"]
        try config.write(paths.support.appendingPathComponent("config.json"))
        let bundled = try BundleCatalogSource(resources: root.appendingPathComponent("Resources"), catalog: catalog)
        let source: any CatalogSource = unavailableSchema ? MenuSchemaFailureSource(base: bundled) : bundled
        let memoryFile = root.appendingPathComponent("memory-fixture.json")
        // 10 GB of room under the 128 GB Mac's line (100/128 of 128,000 MB), plus the 28 GB above it that admission subtracts.
        try WireJSON.encode(["available_mb": 38000]).write(to: memoryFile)
        let service = try await RealModelsService.open(
            catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation,
            source: source, kinds: KindSpec.defaults,
            paths: paths, artifacts: MenuImageStubArtifact(),
            memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": memoryFile.path], totalMB: 128_000))
        try await service.add(id: model.id)
        // An added model starts out of the menu; these tests drive its menu tools, so Show in menu is on.
        try await service.setToolsShown(id: model.id, on: true)
        return service
    }
}

private actor MenuSchemaFailureSource: CatalogSource {
    static let message = "Operation schema source is unavailable."
    let base: BundleCatalogSource
    private(set) var failedReads = 0
    init(base: BundleCatalogSource) { self.base = base }
    func read(path: String, commit: String) async throws -> Data {
        if path == "models/fixture-images/ops/generate.input.json" {
            failedReads += 1
            throw WorkerFailure(code: "fixture_schema", message: Self.message)
        }
        return try await base.read(path: path, commit: commit)
    }
}

private struct MenuImageStubArtifact: WorkerArtifact {
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        let executable = try #require(ProcessInfo.processInfo.environment["ARSENAL_TEST_STUB_WORKER"])
        return WorkerLaunch(
            executable: URL(fileURLWithPath: executable), arguments: ["--kind", model.kind], kind: model.kind, runner: "native",
            environment: ["ARSENAL_SUPPORT_DIR": paths.support.path, "ARSENAL_CACHE_DIR": paths.cache.path, "PATH": "/usr/bin:/bin"],
            log: paths.support.appendingPathComponent("stub.log"))
    }
}

private extension RealModelsService {
    func menuConvertToLocal(_ folder: URL) throws {
        let local = try LocalModelFolders.discoverSnapshot(folder)
        catalog.models.removeAll { $0.id == local.manifest.id }
        localModels[local.manifest.id] = local
    }

    /// Save a real native stub's runnable snapshot as a dormant recovery revision, with a rejected current folder.
    func menuSaveRecovery(_ id: String) async throws {
        let previous = ResidentRevision(
            manifest: try #require(residentManifests[id]), operations: try #require(residentOperations[id]),
            settings: try #require(residentSettings[id]), launch: try #require(residentLaunches[id]), load: try #require(residentLoads[id]),
            preference: try #require(loadedPreferences[id]), resident: try #require(residents[id]), digest: "approved-native", menus: residentMenus[id] ?? [])
        try saveRecoveryRevision(id, previous)
        try await unload(id: id)
        let current = try manifest(id)
        // Recovery belongs to a local model, never simultaneously to the signed catalog.
        catalog.models.removeAll { $0.id == id }
        dormantRevisions[id] = previous
        localModels[id] = LocalModelFolder(manifest: current, directory: paths.support, fileDigests: [:], contentDigest: "rejected")
        rejectedRevisionDigests[id] = "rejected"; config.rejectedRevisions[id] = "rejected"
        #expect(recoveryRevision(id) != nil)
    }
    /// Hold admission state without a download, real model or asynchronous race; HTTP still follows production run.
    func menuAdmissionState(_ id: String, _ state: String) {
        gets[id] = state.hasSuffix("get") ? Task {} : nil
        updating = state.hasSuffix("updating")
    }
}
