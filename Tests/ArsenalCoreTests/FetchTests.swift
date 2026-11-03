// SPDX-License-Identifier: MIT
import Testing
import Foundation
import CryptoKit
import Darwin
@testable import ArsenalCore

// swiftlint:disable:next type_body_length
@Suite final class FetchTests: @unchecked Sendable {
    static let payload = Data(String(repeating: "arsenal-test-weights\n", count: 8192).utf8)
    func temp() throws -> URL {
        let root = URL(fileURLWithPath: "/tmp/arsenal-fetch/tests/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true); return root
    }
    func fixture(_ root: URL, dtype: String = "fp16", extras: String = "", worker: String = "chat") throws -> (URL, ModelManifest) {
        let folder = root.appendingPathComponent("tiny-test")
        for path in ["log", "gate", "bench"] { try FileManager.default.createDirectory(at: folder.appendingPathComponent(path), withIntermediateDirectories: true) }
        for path in ["README.md", "gate/gate.toml", "bench/bench.toml", "log/001-test.md"] { try Data("fixture".utf8).write(to: folder.appendingPathComponent(path)) }
        let text = """
            schema = 1
            id = "tiny-test"
            name = "Tiny Test"
            kind = "chat"
            summary = "Test fixture"
            publisher = "Arsenal"
            released = 2026
            tasks = ["chat"]
            languages = ["en"]
            params = "1"
            \(extras)
            [licence]
            spdx = "MIT"
            commercial = true
            url = "https://example.test/LICENSE"
            [weights]
            repo = "test/good"
            revision = "\(String(repeating: "a", count: 40))"
            dtype = "\(dtype)"
            files = ["weights.bin"]
            [weights.sha256]
            "weights.bin" = "\(FileDigest.sha256(Self.payload))"
            [weights.bytes]
            "weights.bin" = \(Self.payload.count)
            [[runner]]
            name = "native"
            language = "swift"
            worker = "\(worker)"
            protocol = 1
            chips = ["*"]
            min_macos = "14.0"
            fast_exact = true
            [gate]
            recipe = "gate/gate.toml"
            [bench]
            recipe = "bench/bench.toml"
            """
        try text.write(to: folder.appendingPathComponent("model.toml"), atomically: true, encoding: .utf8)
        return (folder, try ModelManifest.load(folder))
    }
    func file(hash: String? = nil) -> DerivedFile { DerivedFile(path: "weights.bin", bytes: Int64(Self.payload.count), sha256: hash ?? FileDigest.sha256(Self.payload)) }
    private func client(_ server: StandIn) -> HuggingFaceClient { var c = HuggingFaceClient(baseURL: server.url); c.deadline = 10; c.token = { nil }; return c }
    @Test func testCorruptAndOversizedPartsDoNotPoisonNextGet() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        for size in [Self.payload.count, Self.payload.count + 1] {
            let directory = root.appendingPathComponent("corrupt-\(size)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let part = directory.appendingPathComponent(file().sha256 + ".part")
            try Data(repeating: 120, count: size).write(to: part)
            let journal = DownloadJournal(
                repo: "test/good", revision: String(repeating: "a", count: 40), path: file().path, sha256: file().sha256, bytes: file().bytes, received: Int64(size),
                etag: "\"fixed\"")
            let journalURL = directory.appendingPathComponent(file().sha256 + ".json")
            try JSONEncoder().encode(journal).write(to: journalURL)
            do {
                _ = try await client(server).download(repo: "test/good", revision: journal.revision, file: file(), directory: directory); Issue.record("corrupt part accepted")
            } catch {}
            #expect(!FileManager.default.fileExists(atPath: part.path))
            #expect(!FileManager.default.fileExists(atPath: journalURL.path))
            let result = try await client(server).download(repo: "test/good", revision: journal.revision, file: file(), directory: directory)
            #expect(try Data(contentsOf: result) == Self.payload)
            let quarantined = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil).filter {
                $0.lastPathComponent.hasPrefix(file().sha256 + ".part.bad-")
            }
            #expect(quarantined.count == 1)
            #expect(try Data(contentsOf: quarantined[0]) == Data(repeating: 120, count: size))
        }
    }
    @Test func testFreshChecksumMismatchCanRetrySuccessfully() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let directory = root.appendingPathComponent("downloads")
        do {
            _ = try await client(server).download(repo: "test/wrong-once", revision: String(repeating: "a", count: 40), file: file(), directory: directory)
            Issue.record("wrong hash accepted")
        } catch {}
        let result = try await client(server).download(repo: "test/wrong-once", revision: String(repeating: "a", count: 40), file: file(), directory: directory)
        #expect(try Data(contentsOf: result) == Self.payload)
    }
    @Test func testDiskFullDuringDownloadRemovesPartialAndRetrySucceeds() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let directory = root.appendingPathComponent("downloads"), revision = String(repeating: "a", count: 40)
        var full = client(server)
        // The disk fills once a write would leave more than 32 KiB on disk; the payload is larger, and the write fails with ENOSPC
        // as the filesystem reports it. The check does not depend on how the loopback server chunks the body.
        #expect(Int64(Self.payload.count) > 32 * 1024)
        full.writeFault = { (onDisk: Int64) -> Error? in onDisk > 32 * 1024 ? POSIXError(.ENOSPC) : nil }
        var thrown: Error?
        do { _ = try await full.downloadVerified(repo: "test/good", revision: revision, file: file(), directory: directory) } catch { thrown = error }
        #expect(thrown != nil, "a full disk completed the download")
        #expect(thrown?.localizedDescription == "disk full; free space and retry Get", "message was: \(thrown?.localizedDescription ?? "none")")
        // The partial and its journal are gone, so the next Get starts clean.
        #expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent(file().sha256 + ".part").path))
        #expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent(file().sha256 + ".json").path))
        // Nothing is published: the store holds no blob for the file.
        let store = BlobStore(support: root.appendingPathComponent("support"))
        #expect(try await store.hasBlob(sha256: file().sha256, bytes: file().bytes) == false)
        // With room again, a retry downloads and verifies the whole file, and that verified download publishes.
        let result = try await client(server).downloadVerified(repo: "test/good", revision: revision, file: file(), directory: directory)
        #expect(try Data(contentsOf: result.url) == Self.payload)
        try await store.promote(result)
        #expect(try await store.hasBlob(sha256: file().sha256, bytes: file().bytes) == true)
    }
    @Test func testRejectedDonorResetsAggregateProgressAndPhase() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, model) = try fixture(root)
        let donor = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 120, count: Self.payload.count).write(to: donor)
        let events = ProgressCapture()
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), client: client(server), localSources: [root.appendingPathComponent("local").path])
        _ = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", progress: { events.add($0) })
        var accumulator = try GetProgressAccumulator(files: [.init(contentID: file().sha256, file: file().path, bytes: file().bytes)])
        var states: [PrecisionState] = []
        for event in events.read() { if let state = accumulator.record(event) { states.append(state) } }
        #expect(states.contains { $0.phase == "downloading" && $0.bytes == 0 })
        #expect(states.last?.phase == "downloading" && states.last?.bytes == file().bytes)
        #expect(try Data(contentsOf: donor) == Data(repeating: 120, count: Self.payload.count))
    }
    @Test func testNetworkRestoreReplacesOldLocalImportProvenance() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, initial) = try fixture(root); var model = initial; model.derive = ["int4": "tiny"]
        let donor = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true); try Self.payload.write(to: donor)
        let fetcher = ModelFetcher(
            support: root.appendingPathComponent("support"), client: client(server), converter: TestConverter(), localSources: [root.appendingPathComponent("local").path])
        _ = try await fetcher.get(model: model, modelDirectory: folder, precision: "int4")
        _ = try await fetcher.store.retainVerifiedDerived(model: model.id, revision: model.weights.revision, sourcePrecision: "fp16", derivedPrecision: "int4", thresholdBytes: 1)
        // Keep the donor untouched, but remove it from the next Get's configured source list.
        let network = ModelFetcher(support: root.appendingPathComponent("support"), client: client(server), converter: TestConverter(), localSources: [])
        _ = try await network.get(model: model, modelDirectory: folder, precision: "fp16")
        let restored = try #require(
            try await network.store.retainVerifiedDerived(model: model.id, revision: model.weights.revision, sourcePrecision: "fp16", derivedPrecision: "int4", thresholdBytes: 1))
        #expect(restored.importedFromLocalCopy == false)
        #expect(restored.localSources == [] && restored.localCopyBytes == 0)
        #expect(restored.note == "16-bit copy removed; Get re-downloads it (0.0 GB)")
        #expect(try Data(contentsOf: donor) == Self.payload)
    }
    @Test func testLocalImportRetentionRecordsLogicalNotPhysicalBytes() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root); var model = initial; model.derive = ["int4": "tiny"]
        let donor = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true); try Self.payload.write(to: donor)
        var offline = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:9")!); offline.maximumAttempts = 1
        let support = root.appendingPathComponent("support")
        let fetcher = ModelFetcher(support: support, client: offline, converter: TestConverter(), localSources: [root.appendingPathComponent("local").path])
        _ = try await fetcher.get(model: model, modelDirectory: folder, precision: "int4")
        // Explicit verification must not erase import provenance.
        _ = try await fetcher.verifyInstalled(model: model, precision: "fp16", forceHash: true)
        let receipt = try #require(
            try await fetcher.store.retainVerifiedDerived(model: model.id, revision: model.weights.revision, sourcePrecision: "fp16", derivedPrecision: "int4", thresholdBytes: 1))
        let json = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(receipt)) as? [String: Any])
        #expect(receipt.note == "16-bit copy removed; Get re-imports it from local copy")
        #expect(json["importedFromLocalCopy"] as? Bool == true)
        #expect(json["localCopyBytes"] as? Int64 == file().bytes)
        #expect(json["localSources"] as? [String] == [donor.path])
        #expect(json["reclaimedBytesMeaning"] as? String == "logical bytes unlinked, not physical disk space saved")
        #expect(try Data(contentsOf: donor) == Self.payload)
        _ = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16") // re-imports offline
    }
    @Test func testTOMLSubsetAndStableWriter() throws {
        let text = """
            title = 'Reader #1' # comment
            number = 1_024
            active = true
            values = [1, 2.5, "x",]
            [pins]
            "config.json" = "hash"
            [[runner]]
            name = "one"
            params = { count = 2, nested = { b = true } }
            [[runner]]
            name = "two"
            [runner.extra]
            source = "hi\\nthere"
            """
        var p = TOMLParser(text); let a = try p.parse(); var q = TOMLParser(TOMLWriter.encode(a)); expectEqual(a, try q.parse())
        var duplicate = TOMLParser("a = 1\na = 2"); expectThrows(try duplicate.parse())
        var invalid = TOMLParser("path = ["); expectThrows(try invalid.parse())
    }
    @Test func testBareValueLineDelimiters() throws {
        for ending in ["\n", "\r\n", " # trailing comment\n", " # trailing comment\r\n"] {
            let text = "integer = 1" + ending + "decimal = 2.5" + ending + "active = true" + ending + "id = \"next\""
            var p = TOMLParser(text); let doc = try p.parse()
            expectEqual(doc["integer"], .integer(1)); expectEqual(doc["decimal"], .decimal(2.5)); expectEqual(doc["active"], .bool(true)); expectEqual(doc["id"], .string("next"))
        }
        for (text, value) in [("n = 1", TOMLValue.integer(1)), ("n = 2.5", .decimal(2.5)), ("n = false", .bool(false))] {
            var p = TOMLParser(text); expectEqual(try p.parse()["n"], value)
        }
        var invalidBare = TOMLParser("n = bare\nid = \"next\""); expectThrows(try invalidBare.parse())
    }
    @Test func testManifestFolderValidationAndDynamicKind() throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let (folder, m) = try fixture(root)
        expectEqual(m.precisions, ["fp16"])
        try FileManager.default.createSymbolicLink(atPath: folder.appendingPathComponent("escape").path, withDestinationPath: "/etc/passwd")
        expectThrows(try ModelManifest.load(folder))
        try FileManager.default.removeItem(at: folder.appendingPathComponent("escape"))
        try FileManager.default.removeItem(at: folder.appendingPathComponent("log/001-test.md")); expectThrows(try ModelManifest.load(folder))
        let kinds = root.appendingPathComponent("kinds"); try FileManager.default.createDirectory(at: kinds, withIntermediateDirectories: true)
        try "id = \"new-kind\"\ntitle = \"New kind\"\norder = 42\nsuite = \"new-v1\"\ncolumns = []\napi = {}".write(
            to: kinds.appendingPathComponent("new-kind.toml"), atomically: true, encoding: .utf8)
        expectEqual(try KindSpec.load(directory: kinds).first?.id, "new-kind")
    }
    @Test func testPublishedModelSchema() throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let (folder, _) = try fixture(root)
        let schema = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent(
            "Resources/schemas/model.schema.json")
        var parser = TOMLParser(try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8)); var doc = try parser.parse()
        let data = try Data(contentsOf: schema); expectTrue(try ModelSchemaValidation.validate(document: doc, schemaData: data).isEmpty)
        doc["context"] = .integer(-1); expectFalse(try ModelSchemaValidation.validate(document: doc, schemaData: data).isEmpty)
        doc["context"] = .integer(32)
        if case .table(var weights) = doc["weights"] { weights["dtype"] = .string("fp32"); doc["weights"] = .table(weights) }
        expectFalse(try ModelSchemaValidation.validate(document: doc, schemaData: data).isEmpty)
    }
    @Test func testCatalogSignatureTamperAndDowngrade() throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let (_, m) = try fixture(root)
        let doc = CatalogDocument(
            sequence: 2, sourceCommit: String(repeating: "a", count: 40),
            models: [CatalogModel(manifest: m, files: [.init(path: "model.toml", sha256: String(repeating: "a", count: 64), bytes: 4)])])
        let key = Curve25519.Signing.PrivateKey(); let data = try doc.encoded(); let signature = try key.signature(for: data)
        expectEqual(try CatalogTrust.verify(data: data, signature: signature, publicKey: key.publicKey.rawRepresentation).sequence, 2)
        expectThrows(try CatalogTrust.verify(data: data + Data([32]), signature: signature, publicKey: key.publicKey.rawRepresentation))
        expectThrows(try CatalogTrust.verify(data: data, signature: signature, publicKey: key.publicKey.rawRepresentation, minimumSequence: 3))
        var bad = doc; bad.models[0].files[0].path = "../escape"; expectThrows(try bad.validate())
    }
    @Test func testGitAndHTTPFolderInstallOfflineAndUnlisted() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let models = root.appendingPathComponent("models"); try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        let (folder, manifest) = try fixture(models)
        _ = try GitRead.run(repo: root, args: ["init", "-q"]); _ = try GitRead.run(repo: root, args: ["add", "models"])
        _ = try GitRead.run(repo: root, args: ["-c", "user.name=Tests", "-c", "user.email=test@example.test", "commit", "-qm", "fixture"])
        let commit = String(data: try GitRead.run(repo: root, args: ["rev-parse", "HEAD"]), encoding: .utf8)!.trimmingCharacters(in: .whitespacesAndNewlines)
        let files = try SafePath.files(in: folder).map { path in
            CatalogFile(path: path, sha256: try FileDigest.sha256(folder.appendingPathComponent(path)), bytes: try FileDigest.bytes(folder.appendingPathComponent(path)))
        }
        let catalog = CatalogDocument(sequence: 2, sourceCommit: commit, models: [CatalogModel(manifest: manifest, files: files)])
        let key = Curve25519.Signing.PrivateKey(); let data = try catalog.encoded(); let signature = try key.signature(for: data)
        let installer = ModelFolderInstaller(support: root.appendingPathComponent("support"))
        try await installer.accept(data: data, signature: signature, publicKey: key.publicKey.rawRepresentation)
        let installed = try await installer.install(id: manifest.id, source: LocalGitCatalogSource(repo: root)); expectEqual(try ModelManifest.load(installed), manifest)
        let verified = try await installer.verify(installedFolder: installed); expectEqual(verified, manifest)
        let server = try await StandIn(root: root); defer { server.stop() }
        let httpInstaller = ModelFolderInstaller(support: root.appendingPathComponent("http-support"))
        try await httpInstaller.accept(data: data, signature: signature, publicKey: key.publicKey.rawRepresentation)
        _ = try await httpInstaller.install(id: manifest.id, source: HTTPSCatalogSource(rawBase: server.url))
        let reopened = ModelFolderInstaller(support: root.appendingPathComponent("support")); let restored = try await reopened.restore(publicKey: key.publicKey.rawRepresentation)
        expectEqual(restored, catalog)
        _ = try await reopened.install(id: manifest.id, source: LocalGitCatalogSource(repo: URL(fileURLWithPath: "/no-repo")))
        try Data("unlisted".utf8).write(to: installed.appendingPathComponent("unlisted.txt"))
        do { _ = try await reopened.verify(installedFolder: installed); Issue.record("non-fetching verification accepted changed folder") } catch {}
        do { _ = try await reopened.install(id: manifest.id, source: LocalGitCatalogSource(repo: root)); Issue.record("unlisted file accepted") } catch {}
        var older = catalog; older.sequence = 1; let oldData = try older.encoded()
        do {
            _ = try await reopened.accept(data: oldData, signature: key.signature(for: oldData), publicKey: key.publicKey.rawRepresentation); Issue.record("downgrade accepted")
        } catch {}
    }
    @Test func catalogKindSourceReadsCurrentAndHistoricalLayouts() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try GitRead.run(repo: root, args: ["init", "-q"])
        let old = root.appendingPathComponent("kinds/fixture.toml"), current = root.appendingPathComponent("Resources/kinds/fixture.toml")
        try FileManager.default.createDirectory(at: old.deletingLastPathComponent(), withIntermediateDirectories: true)
        let bytes = Data("id = \"fixture\"".utf8)
        try bytes.write(to: old)
        _ = try GitRead.run(repo: root, args: ["add", "."])
        _ = try GitRead.run(repo: root, args: ["-c", "user.name=Tests", "-c", "user.email=tests@example.invalid", "commit", "-qm", "historical kinds"])
        let oldCommit = String(decoding: try GitRead.run(repo: root, args: ["rev-parse", "HEAD"]), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        let server = try await StandIn(root: root); defer { server.stop() }
        let local = LocalGitCatalogSource(repo: root), remote = HTTPSCatalogSource(rawBase: server.url)
        #expect(try await local.read(path: "kinds/fixture.toml", commit: oldCommit) == bytes)
        #expect(try await remote.read(path: "kinds/fixture.toml", commit: oldCommit) == bytes)
        try FileManager.default.createDirectory(at: current.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: old, to: current)
        _ = try GitRead.run(repo: root, args: ["add", "-A"])
        _ = try GitRead.run(repo: root, args: ["-c", "user.name=Tests", "-c", "user.email=tests@example.invalid", "commit", "-qm", "current kinds"])
        let commit = String(decoding: try GitRead.run(repo: root, args: ["rev-parse", "HEAD"]), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(try await local.read(path: "kinds/fixture.toml", commit: commit) == bytes)
        #expect(try await local.read(path: "kinds/fixture.toml", commit: oldCommit) == bytes)
        #expect(try await remote.read(path: "kinds/fixture.toml", commit: commit) == bytes)
    }

    @Test func testInterruptedDownloadResumesAndVerifies() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let path = try await client(server).download(
            repo: "test/interrupted", revision: String(repeating: "a", count: 40), file: file(), directory: root.appendingPathComponent("downloads"))
        expectEqual(try Data(contentsOf: path), Self.payload)
    }
    func seed(_ root: URL, mode: String) throws -> URL {
        let dir = root.appendingPathComponent(mode); try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let hash = file().sha256
        try Self.payload.prefix(1024).write(to: dir.appendingPathComponent(hash + ".part"))
        let journal = DownloadJournal(
            repo: "test/\(mode)", revision: String(repeating: "a", count: 40), path: "weights.bin", sha256: hash, bytes: Int64(Self.payload.count), received: 1024,
            etag: "\"fixed\"")
        try JSONEncoder().encode(journal).write(to: dir.appendingPathComponent(hash + ".json")); return dir
    }
    @Test func testCrashJournalResumeAnd200Restarts() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        for mode in ["good", "restart"] {
            let directory = try seed(root, mode: mode)
            let path = try await client(server).download(repo: "test/\(mode)", revision: String(repeating: "a", count: 40), file: file(), directory: directory)
            expectEqual(try Data(contentsOf: path), Self.payload)
        }
    }
    @Test func testWrongHashBadRangeAndDeniedFail() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        for mode in ["wrong", "bad-range", "denied"] {
            let directory = mode == "bad-range" ? try seed(root, mode: mode) : root.appendingPathComponent(mode)
            do {
                _ = try await client(server).download(repo: "test/\(mode)", revision: String(repeating: "a", count: 40), file: file(), directory: directory)
                Issue.record("\(mode) accepted")
            } catch { expectFalse(error.localizedDescription.isEmpty) }
        }
    }
    @Test func testCancellationLeavesResumableJournal() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let c = client(server); let file = file(); let dir = root.appendingPathComponent("downloads")
        let task = Task { try await c.download(repo: "test/slow", revision: String(repeating: "a", count: 40), file: file, directory: dir) }
        try await Task.sleep(for: .milliseconds(150)); task.cancel()
        do { _ = try await task.value; Issue.record("cancel ignored") } catch { expectTrue(error is CancellationError) }
        expectTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent(file.sha256 + ".json").path))
        let result = try await c.download(repo: "test/slow", revision: String(repeating: "a", count: 40), file: file, directory: dir)
        expectEqual(try Data(contentsOf: result), Self.payload)
    }
    @Test func testBlobDedupReferenceCountsAndResidentLease() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let store = BlobStore(support: root)
        let f = file(); let part = root.appendingPathComponent("part"); try Self.payload.write(to: part); try await store.promote(part, sha256: f.sha256, bytes: f.bytes)
        for model in ["one", "two"] {
            _ = try await store.commitSnapshot(.init(model: model, revision: String(repeating: "a", count: 40), precision: "fp16", files: [f], derived: nil))
        }
        let counts = try await store.referenceCounts(); expectEqual(counts[f.sha256], 2)
        try await store.delete(model: "one"); let exists = try await store.hasBlob(sha256: f.sha256, bytes: f.bytes); expectTrue(exists)
        await store.retain(model: "two")
        do { try await store.delete(model: "two"); Issue.record("resident delete allowed") } catch {}
        await store.release(model: "two"); try await store.delete(model: "two")
        let blob = try await store.blobURL(sha256: f.sha256); expectFalse(FileManager.default.fileExists(atPath: blob.path))
    }
    @Test func streamedHalfCastPinsConvertedBytesAndDropsOriginal() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root, dtype: "bf16")
        var model = initial; model.derive = ["fp16": "cast:flux2-klein-4b", "int8": "tiny"]
        let donor = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.payload.write(to: donor)
        let support = root.appendingPathComponent("support")
        let fetcher = ModelFetcher(support: support, converter: TestConverter(), localSources: [root.appendingPathComponent("local").path])
        #expect(model.basePrecision == "fp16" && model.source(for: "fp16") == nil)
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", approvedDownloads: [:])
        #expect(try Data(contentsOf: result.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data("fp16".utf8))
        let snapshot = try #require(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "fp16", forceHash: true))
        #expect(snapshot.derived?.sourceDtype == "bf16")
        #expect(snapshot.derived?.sourceFiles == [file()])
        #expect(try await fetcher.store.hasBlob(sha256: file().sha256, bytes: file().bytes) == false)
        #expect(try Data(contentsOf: donor) == Self.payload)
        let quantized = try await fetcher.get(model: model, modelDirectory: folder, precision: "int8", approvedDownloads: [:])
        #expect(try Data(contentsOf: quantized.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data("fp16int8".utf8))
        #expect(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "int8", forceHash: true) != nil)
        #expect(try await fetcher.updateRequired(model: model, precision: "int8") == false)
        var changedPins = model
        changedPins.weights.sha256?["weights.bin"] = String(repeating: "c", count: 64)
        #expect(try await fetcher.updateRequired(model: changedPins, precision: "int8"))
        #expect(try FileManager.default.contentsOfDirectory(atPath: support.appendingPathComponent("weights").path).allSatisfy { !$0.hasPrefix(".cast-") })
    }
    @Test func fp32SourceCanSelectBF16AndDeriveQuantsWithoutRetainingOriginal() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let extras = """
            [derive]
            bf16 = "cast:qualified-bf16"
            int8 = "tiny"
            [[format]]
            id = "bf16"
            bits_per_weight = 16
            storage = "bf16"
            source = "derived"
            runners = ["native"]
            """
        let (folder, model) = try fixture(root, dtype: "fp32", extras: extras)
        let donor = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.payload.write(to: donor)
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), converter: TestConverter(), localSources: [root.appendingPathComponent("local").path])
        #expect(model.basePrecision == "bf16" && model.source(for: "bf16") == nil && !model.precisions.contains("fp16"))
        let base = try await fetcher.get(model: model, modelDirectory: folder, precision: "bf16", approvedDownloads: [:])
        #expect(try Data(contentsOf: base.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data("bf16".utf8))
        let native = try await fetcher.verifyInstalled(model: model, precision: "bf16", forceHash: true)
        #expect(native.weightsDirectory.standardizedFileURL.resolvingSymlinksInPath().path == base.weightsDirectory.standardizedFileURL.resolvingSymlinksInPath().path)
        let snapshot = try #require(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "bf16", forceHash: true))
        #expect(snapshot.derived?.sourceDtype == "fp32")
        #expect(try await fetcher.store.hasBlob(sha256: file().sha256, bytes: file().bytes) == false)
        let quantized = try await fetcher.get(model: model, modelDirectory: folder, precision: "int8", approvedDownloads: [:])
        #expect(try Data(contentsOf: quantized.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data("bf16int8".utf8))
        #expect(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "int8", forceHash: true)?.derived?.sourceDtype == "bf16")
        #expect(try Data(contentsOf: donor) == Self.payload)
    }

    @Test(arguments: [false, true]) func multiFileCastFailureOrCancellationCleansStaging(cancel: Bool) async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root, dtype: "bf16")
        var model = initial
        model.derive = ["fp16": "cast:flux2-klein-4b"]
        model.weights.files = ["a.bin", "b.bin"]
        model.weights.sha256 = Dictionary(uniqueKeysWithValues: model.weights.files.map { ($0, FileDigest.sha256(Self.payload)) })
        model.weights.bytes = Dictionary(uniqueKeysWithValues: model.weights.files.map { ($0, Int64(Self.payload.count)) })
        for name in model.weights.files {
            let donor = root.appendingPathComponent("local/test/good/" + name)
            try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Self.payload.write(to: donor)
        }
        let support = root.appendingPathComponent("support")
        let fetcher = ModelFetcher(support: support, converter: InterruptingCastConverter(cancel: cancel), localSources: [root.appendingPathComponent("local").path])
        let target = model
        let task = Task { try await fetcher.get(model: target, modelDirectory: folder, precision: "fp16", approvedDownloads: [:]) }
        do { _ = try await task.value; Issue.record("Interrupted cast unexpectedly committed") } catch {
            if cancel { #expect(error is CancellationError) }
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: support.appendingPathComponent("weights").path).allSatisfy { !$0.hasPrefix(".cast-") })
        #expect(try await fetcher.store.hasBlob(sha256: FileDigest.sha256(Self.payload), bytes: Int64(Self.payload.count)) == false)
        let converted = Self.payload + Data("fp16".utf8)
        #expect(try await fetcher.store.hasBlob(sha256: FileDigest.sha256(converted), bytes: Int64(converted.count)) == false)
        #expect(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "fp16") == nil)
    }
    @Test(arguments: ["stable-audio-3-medium", "stable-audio-3-small-sfx"]) func recipeUpgradeOnlyRewritesSnapshotsOnExplicitGet(modelID: String) async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root)
        let donor = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.payload.write(to: donor)
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), converter: TestConverter(), localSources: [root.appendingPathComponent("local").path])
        let old = try await fetcher.get(model: initial, modelDirectory: folder, precision: "fp16", approvedDownloads: [:])
        let receipt = old.weightsDirectory.appendingPathComponent("arsenal-snapshot.json")
        let before = try Data(contentsOf: receipt)
        var upgraded = initial; upgraded.derive = ["fp16": "cast:" + modelID, "int8": "intake-v1:" + modelID]
        #expect(try await fetcher.updateRequired(model: upgraded, precision: "fp16"))
        do { _ = try await fetcher.verifyInstalled(model: upgraded, precision: "fp16"); Issue.record("Old uncast snapshot accepted") } catch {}
        #expect(try Data(contentsOf: receipt) == before)
        #expect(try Data(contentsOf: old.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload)
        let next = try await fetcher.get(model: upgraded, modelDirectory: folder, precision: "int8", approvedDownloads: [:])
        #expect(try Data(contentsOf: next.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data("fp16int8".utf8))
        #expect(try await fetcher.updateRequired(model: upgraded, precision: "int8") == false)
        #expect(try Data(contentsOf: donor) == Self.payload)
    }
    /// A BF16 source whose base is a model-owned `cast:` recipe (G3's audio folders): Get stages the original pinned
    /// source, normalizes it file by file, calls the model converter once on the half source, and never recurses into
    /// its own base precision.
    @Test func bf16SourceWithModelOwnedBaseCastStagesTheOriginalOnce() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root, dtype: "bf16", worker: "audio")
        var model = initial
        model.precision = NativeHalfPolicy(dtype: "fp16")
        model.derive = ["fp16": "cast:audio-g3:fixture", "int8": "audio-affine-v1:fixture"]
        #expect(model.normalizesNativeHalf && model.basePrecision == "fp16" && model.conversionRecipe(for: "fp16") == "cast:audio-g3:fixture")
        #expect(model.source(for: "fp16") == nil && model.problems(folder: folder).isEmpty)
        let donor = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.payload.write(to: donor)
        let support = root.appendingPathComponent("support")
        let converter = RecordingConverter()
        let fetcher = ModelFetcher(support: support, converter: converter, localSources: [root.appendingPathComponent("local").path])
        let base = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", approvedDownloads: [:])
        // Normalization (one source file), then the model-owned recipe on the normalized half source.
        #expect(converter.calls.map(\.recipe) == [NativeHalfPolicy.recipe, "cast:audio-g3:fixture"])
        #expect(converter.calls.map(\.source) == [Self.payload, Self.payload + Data("fp16".utf8)])
        #expect(try Data(contentsOf: base.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data("fp16fp16".utf8))
        let snapshot = try #require(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "fp16", forceHash: true))
        #expect(snapshot.derived?.sourceDtype == "bf16" && snapshot.derived?.recipe == "cast:audio-g3:fixture")
        #expect(snapshot.derived?.normalization == model.nativeHalfNormalization && snapshot.derived?.sourceFiles == [file()])
        #expect(try await fetcher.store.hasBlob(sha256: file().sha256, bytes: file().bytes) == false)
        #expect(try Data(contentsOf: donor) == Self.payload)
        // A derived tier starts from the committed base and never re-normalizes or re-casts.
        let quantized = try await fetcher.get(model: model, modelDirectory: folder, precision: "int8", approvedDownloads: [:])
        #expect(converter.calls.map(\.recipe) == [NativeHalfPolicy.recipe, "cast:audio-g3:fixture", "audio-affine-v1:fixture"])
        #expect(try Data(contentsOf: quantized.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data("fp16fp16int8".utf8))
        #expect(try await fetcher.updateRequired(model: model, precision: "int8") == false)
        #expect(try FileManager.default.contentsOfDirectory(atPath: support.appendingPathComponent("weights").path).allSatisfy { !$0.hasPrefix(".cast-") })
    }
    /// `[precision] source = "raw"` (LaMa's BatchNorm fold): Get skips the half normalization and hands the model's own
    /// cast the verified float32 source once; the snapshot records the raw-cast provenance and the source is collected.
    @Test func rawSourceCastSkipsHalfNormalization() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root, dtype: "fp32", worker: "restoration")
        var model = initial
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("log"), withIntermediateDirectories: true)
        try Data("fold evidence\n".utf8).write(to: folder.appendingPathComponent("log/001-port.md"))
        model.precision = NativeHalfPolicy(dtype: "fp16", reason: "BatchNorm folding needs full range.", evidence: "log/001-port.md", source: "raw")
        model.derive = ["fp16": "cast:restoration-fixture"]
        #expect(model.castsRawSource && model.nativeHalfNormalization == "raw-cast-v1:fp16" && model.problems(folder: folder).isEmpty)
        var unreasoned = model; unreasoned.precision?.reason = nil
        #expect(unreasoned.problems(folder: folder).contains { $0.contains("a raw source needs precision.reason") })
        var shared = model; shared.derive = ["fp16": NativeHalfPolicy.recipe]
        #expect(shared.problems(folder: folder).contains { $0.contains("model's own cast: recipe") })
        let donor = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.payload.write(to: donor)
        let converter = RecordingConverter()
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), converter: converter, localSources: [root.appendingPathComponent("local").path])
        let base = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", approvedDownloads: [:])
        #expect(converter.calls.map(\.recipe) == ["cast:restoration-fixture"])
        #expect(converter.calls.map(\.source) == [Self.payload])
        #expect(try Data(contentsOf: base.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data("fp16".utf8))
        let snapshot = try #require(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "fp16", forceHash: true))
        #expect(snapshot.derived?.sourceDtype == "fp32" && snapshot.derived?.normalization == "raw-cast-v1:fp16")
        #expect(try await fetcher.store.hasBlob(sha256: file().sha256, bytes: file().bytes) == false)
        #expect(try await fetcher.updateRequired(model: model, precision: "fp16") == false)
    }
    /// SAM 3.1's FP32 source holds an unused tensor beyond FP16's range: the generic half normalization refused it at
    /// Get (22 Oct). It declares a raw source, so its own recipe receives the original file unnormalized, once, and
    /// drops that tensor (the shape models/sam3.1/model.toml ships: cast:sam31-fp16-v3, [precision] source = "raw").
    @Test func sam31CastReadsTheOriginalSourceWithoutNormalization() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root, dtype: "fp32", worker: "perception")
        var model = initial
        model.derive = ["fp16": "cast:sam31-fp16-v3"]
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("log"), withIntermediateDirectories: true)
        try Data("cast evidence\n".utf8).write(to: folder.appendingPathComponent("log/001-port.md"))
        model.precision = NativeHalfPolicy(dtype: "fp16", reason: "An unused tensor exceeds FP16's range.", evidence: "log/001-port.md", source: "raw")
        #expect(model.normalizesNativeHalf && model.castsRawSource && model.basePrecision == "fp16" && model.problems(folder: folder).isEmpty)
        let donor = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.payload.write(to: donor)
        let converter = RecordingConverter()
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), converter: converter, localSources: [root.appendingPathComponent("local").path])
        let base = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", approvedDownloads: [:])
        #expect(converter.calls.map(\.recipe) == ["cast:sam31-fp16-v3"])
        #expect(converter.calls.map(\.source) == [Self.payload])
        #expect(try Data(contentsOf: base.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data("fp16".utf8))
        let snapshot = try #require(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "fp16", forceHash: true))
        #expect(snapshot.derived?.sourceDtype == "fp32" && snapshot.derived?.recipe == "cast:sam31-fp16-v3")
    }
    /// An auxiliary repository declared `into` a snapshot directory (SAM Audio's T5 encoder) is converted once at Get
    /// into the base snapshot, never committed as its own snapshot, and not fetched again for a derived tier.
    @Test func incorporatedAssetIsStoredOnceInsideTheBaseSnapshot() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root, dtype: "bf16", worker: "audio")
        let encoder = Data(String(repeating: "t5-encoder\n", count: 512).utf8)
        var model = initial
        model.precision = NativeHalfPolicy(dtype: "fp16")
        model.derive = ["fp16": "cast:component-fixture", "int8": "component-fixture"]
        model.asset = [
            AssetSpec(
                id: "t5", repo: "test/t5", repoType: "model", revision: initial.weights.revision, files: ["model.bin"], licence: .spdx("MIT"),
                sha256: ["model.bin": FileDigest.sha256(encoder)], bytes: ["model.bin": Int64(encoder.count)], into: "text")
        ]
        #expect(model.problems(folder: folder).isEmpty && model.incorporatedAssets.map(\.id) == ["t5"] && model.separateAssets.isEmpty)
        for (repo, name, data) in [("test/good", "weights.bin", Self.payload), ("test/t5", "model.bin", encoder)] {
            let donor = root.appendingPathComponent("local/" + repo + "/" + name)
            try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: donor)
        }
        let support = root.appendingPathComponent("support")
        let fetcher = ModelFetcher(support: support, converter: ComponentConverter(), localSources: [root.appendingPathComponent("local").path])
        let base = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", approvedDownloads: [:])
        #expect(base.assets.isEmpty)
        #expect(try Data(contentsOf: base.weightsDirectory.appendingPathComponent("text/model.bin")) == encoder + Data("fp16".utf8))
        let snapshot = try #require(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "fp16", forceHash: true))
        #expect(snapshot.derived?.assetSources?["t5"] == model.asset?[0].weights)
        // No separate asset snapshot, and the original encoder blob is collected after conversion.
        #expect(!FileManager.default.fileExists(atPath: support.appendingPathComponent("weights/snapshots/asset-t5").path))
        #expect(try await fetcher.store.hasBlob(sha256: FileDigest.sha256(encoder), bytes: Int64(encoder.count)) == false)
        #expect(try await fetcher.verifyInstalled(model: model, precision: "fp16", forceHash: true).assets.isEmpty)
        // The derived tier packs the stored component from the base; it needs no asset source.
        try FileManager.default.removeItem(at: root.appendingPathComponent("local/test/t5"))
        let quantized = try await fetcher.get(model: model, modelDirectory: folder, precision: "int8", approvedDownloads: [:])
        #expect(try Data(contentsOf: quantized.weightsDirectory.appendingPathComponent("text/model.bin")) == encoder + Data("fp16int8".utf8))
        #expect(try await fetcher.updateRequired(model: model, precision: "int8") == false)
        #expect(try FileManager.default.contentsOfDirectory(atPath: support.appendingPathComponent("weights").path).allSatisfy { !$0.hasPrefix(".asset-") })
        // A conversion that drops the declared component is refused.
        try encoder.write(to: root.appendingPathComponent("local/test/t5/model.bin").createParent())
        var dropped = model; dropped.asset?[0].into = "missing"
        let other = ModelFetcher(support: root.appendingPathComponent("other"), converter: ComponentConverter(), localSources: [root.appendingPathComponent("local").path])
        do {
            _ = try await other.get(model: dropped, modelDirectory: folder, precision: "fp16", approvedDownloads: [:]); Issue.record("A dropped component was accepted")
        } catch { #expect("\(error)".contains("did not store asset t5")) }
    }
    @Test func mixedFP16SourceCanHaveAnExplicitCastRecipe() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root)
        var model = initial; model.derive = ["fp16": "cast:stable-audio-3-medium"]
        #expect(model.problems(folder: folder).isEmpty)
        #expect(model.source(for: "fp16") == nil)
    }
    @Test func testGetDerivedFp32AndAssets() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, initial) = try fixture(root)
        var model = initial; model.weights.dtype = "fp32"; model.derive = ["fp16": "cast:tiny", "int8": "opaque:tiny"]
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), client: client(server), converter: TestConverter())
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "int8")
        let snapshot = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: result.weightsDirectory.appendingPathComponent("arsenal-snapshot.json")))
        expectEqual(snapshot.derived?.sourceDtype, "fp16"); expectEqual(snapshot.derived?.recipe, "opaque:tiny")
        let fp16 = try await fetcher.store.snapshotURL(model: model.id, revision: model.weights.revision, precision: "fp16")
        let cast = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: fp16.appendingPathComponent("arsenal-snapshot.json")))
        expectEqual(cast.derived?.sourceHashes["weights.bin"], file().sha256); expectFalse(model.precisions.contains("fp32"))
        let sourceBlob = try await fetcher.store.blobURL(sha256: file().sha256); expectFalse(FileManager.default.fileExists(atPath: sourceBlob.path))
        expectFalse((try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("support/weights").path)).contains { $0.hasPrefix(".cast-") })
    }
    @Test func testVendorDownloadAndSharedDatasetAssetSizes() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, initial) = try fixture(root)
        var model = initial
        model.weights.repo = "test/denied"
        var vendor = initial.weights; vendor.dtype = "int4"; model.vendor = ["int4": vendor]
        model.assets = ["shared"]
        model.asset = [
            AssetSpec(
                id: "shared", repo: "test/good", repoType: "dataset", revision: initial.weights.revision, files: initial.weights.files, licence: .spdx("MIT"),
                sha256: initial.weights.sha256, bytes: initial.weights.bytes)
        ]
        expectEqual(model.downloadBytes(precision: "int4"), Int64(Self.payload.count))
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), client: client(server))
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "int4")
        expectEqual(try Data(contentsOf: result.weightsDirectory.appendingPathComponent("weights.bin")), Self.payload)
        expectTrue(result.assets["shared"] != nil)
        let counts = try await fetcher.store.referenceCounts(); expectEqual(counts[file().sha256], 2)
        // Repeated Get needs no network and reuses the same asset snapshot.
        server.stop()
        let again = try await fetcher.get(model: model, modelDirectory: folder, precision: "int4")
        expectEqual(again.assets.mapValues(\.path), result.assets.mapValues(\.path))
        var unpinned = model; unpinned.vendor?["int4"]?.sha256 = nil; expectEqual(unpinned.downloadBytes(precision: "int4"), nil)
        try await fetcher.store.delete(model: model.id)
        let assetDir = result.assets["shared"]!; expectFalse(FileManager.default.fileExists(atPath: assetDir.path))
        let blobGone = try await fetcher.store.hasBlob(sha256: file().sha256, bytes: file().bytes); expectFalse(blobGone)
    }
    @Test func testQueuedGetCancellationAndReservedBlobs() async throws {
        let gate = GetSerialGate(); let first = UUID(); let second = UUID()
        try await gate.acquire(first)
        let waiting = Task { try await gate.acquire(second) }
        try await Task.sleep(for: .milliseconds(30)); waiting.cancel()
        do { try await waiting.value; Issue.record("queued cancellation ignored") } catch { expectTrue(error is CancellationError) }
        await gate.release(first)
        let next = UUID(); try await gate.acquire(next); await gate.release(next)
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let store = BlobStore(support: root)
        let f = file(); let part = root.appendingPathComponent("part"); try Self.payload.write(to: part)
        await store.reserve(files: [f]); try await store.promote(part, sha256: f.sha256, bytes: f.bytes)
        try await store.delete(model: "unrelated")
        let preserved = try await store.hasBlob(sha256: f.sha256, bytes: f.bytes); expectTrue(preserved)
        // An unreferenced blob is an orphan: an unrelated delete never sweeps it, only deleteOrphans does.
        await store.unreserve(files: [f]); try await store.delete(model: "unrelated")
        let kept = try await store.hasBlob(sha256: f.sha256, bytes: f.bytes); expectTrue(kept)
        expectEqual(try await store.deleteOrphans(), f.bytes)
        let absent = try await store.hasBlob(sha256: f.sha256, bytes: f.bytes); expectFalse(absent)
    }
    @Test func testVerifiedSourceRetentionKeepsSmallReservedAndSharedBlobs() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, initial) = try fixture(root); var model = initial; model.derive = ["int4": "tiny"]
        let support = root.appendingPathComponent("support")
        let fetcher = ModelFetcher(support: support, client: client(server), converter: TestConverter())
        _ = try await fetcher.get(model: model, modelDirectory: folder, precision: "int4")
        let revision = model.weights.revision
        let source = try await fetcher.store.snapshotURL(model: model.id, revision: revision, precision: "fp16")
        let kept = try await fetcher.store.retainVerifiedDerived(model: model.id, revision: revision, sourcePrecision: "fp16", derivedPrecision: "int4")
        expectEqual(kept, nil); expectTrue(FileManager.default.fileExists(atPath: source.path))
        let boundary = try await fetcher.store.retainVerifiedDerived(
            model: model.id, revision: revision, sourcePrecision: "fp16", derivedPrecision: "int4", thresholdBytes: file().bytes)
        expectEqual(boundary, nil)
        do {
            _ = try await fetcher.store.retainVerifiedDerived(model: model.id, revision: revision, sourcePrecision: "fp16", derivedPrecision: "fp16", thresholdBytes: 1)
            Issue.record("selected source deletion accepted")
        } catch {}
        await fetcher.store.reserve(files: [file()])
        let reserved = try await fetcher.store.retainVerifiedDerived(model: model.id, revision: revision, sourcePrecision: "fp16", derivedPrecision: "int4", thresholdBytes: 1)
        expectEqual(reserved, nil); await fetcher.store.unreserve(files: [file()])
        _ = try await fetcher.store.commitSnapshot(SnapshotManifest(model: "other", revision: revision, precision: "fp16", files: [file()], derived: nil))
        let receipt = try await fetcher.store.retainVerifiedDerived(model: model.id, revision: revision, sourcePrecision: "fp16", derivedPrecision: "int4", thresholdBytes: 1)
        expectEqual(receipt?.bytes, file().bytes); expectEqual(receipt?.reclaimedBytes, 0); expectEqual(receipt?.retainedSharedBytes, file().bytes)
        expectEqual(receipt?.sourceFiles, [file()]); expectFalse(FileManager.default.fileExists(atPath: source.path))
        let present = try await fetcher.store.hasBlob(sha256: file().sha256, bytes: file().bytes); expectTrue(present)
        let restored = try await fetcher.store.sourceRetentionReceipt(model: model.id, revision: revision, sourcePrecision: "fp16"); expectEqual(restored, receipt)
        let log = try String(contentsOf: support.appendingPathComponent("logs/store-deletions.jsonl"), encoding: .utf8)
        let first = try #require(JSONSerialization.jsonObject(with: Data(log.components(separatedBy: .newlines)[0].utf8)) as? [String: Any])
        expectEqual(first["path"] as? String, source.path); expectEqual(first["bytes"] as? Int64, file().bytes)
        expectEqual(first["reason"] as? String, "verified_derive_source_retention")
        // The shared source blob permits recreating the removed snapshot offline.
        server.stop(); _ = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16")
        let hidden = try await fetcher.store.sourceRetentionReceipt(model: model.id, revision: revision, sourcePrecision: "fp16"); expectEqual(hidden, nil)
    }
    @Test func testPrunedSourceRedownloadResumesAndInt8NeverUsesInt4() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, initial) = try fixture(root); var model = initial; model.derive = ["int4": "tiny", "int8": "tiny"]
        model.weights.repo = "test/required-range"
        let support = root.appendingPathComponent("support"); let downloads = support.appendingPathComponent("weights/downloads")
        func partial() throws {
            try FileManager.default.createDirectory(at: downloads, withIntermediateDirectories: true)
            try Self.payload.prefix(1024).write(to: downloads.appendingPathComponent(file().sha256 + ".part"))
            let journal = DownloadJournal(
                repo: model.weights.repo, revision: model.weights.revision, path: file().path, sha256: file().sha256, bytes: file().bytes, received: 1024, etag: "\"fixed\"")
            try JSONEncoder().encode(journal).write(to: downloads.appendingPathComponent(file().sha256 + ".json"))
        }
        try partial()
        let fetcher = ModelFetcher(support: support, client: client(server), converter: TestConverter())
        let int4 = try await fetcher.get(model: model, modelDirectory: folder, precision: "int4")
        let receipt = try await fetcher.store.retainVerifiedDerived(
            model: model.id, revision: model.weights.revision, sourcePrecision: "fp16", derivedPrecision: "int4", thresholdBytes: 1)
        expectEqual(receipt?.reclaimedBytes, file().bytes)
        let gone = try await fetcher.store.hasBlob(sha256: file().sha256, bytes: file().bytes); expectFalse(gone)
        expectTrue(FileManager.default.fileExists(atPath: int4.weightsDirectory.path))
        try partial() // This server refuses requests without Range; a later Get must resume, not discard its journal.
        let int8 = try await fetcher.get(model: model, modelDirectory: folder, precision: "int8")
        expectEqual(try Data(contentsOf: int8.weightsDirectory.appendingPathComponent("weights.bin")), Self.payload + Data("int8".utf8))
        let snapshot = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: int8.weightsDirectory.appendingPathComponent("arsenal-snapshot.json")))
        expectEqual(snapshot.derived?.sourceDtype, "fp16"); expectEqual(snapshot.derived?.sourceFiles, [file()])
        expectEqual(snapshot.derived?.sourceRepository, model.weights.repo)
    }
    @Test func testRetentionFailsClosedWithoutRecoveryPinsOrDeletionLog() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, initial) = try fixture(root); var model = initial; model.derive = ["int4": "tiny"]
        let support = root.appendingPathComponent("support"); let fetcher = ModelFetcher(support: support, client: client(server), converter: TestConverter())
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "int4")
        let snapshotURL = result.weightsDirectory.appendingPathComponent("arsenal-snapshot.json")
        let original = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: snapshotURL))
        var legacy = original; legacy.derived?.sourceFiles = nil
        try JSONEncoder().encode(legacy).write(to: snapshotURL)
        try JSONEncoder().encode(legacy.derived!).write(to: result.weightsDirectory.appendingPathComponent("arsenal-derived.json"))
        let kept = try await fetcher.store.retainVerifiedDerived(
            model: model.id, revision: model.weights.revision, sourcePrecision: "fp16", derivedPrecision: "int4", thresholdBytes: 1)
        expectEqual(kept, nil)
        try JSONEncoder().encode(original).write(to: snapshotURL)
        try JSONEncoder().encode(original.derived!).write(to: result.weightsDirectory.appendingPathComponent("arsenal-derived.json"))
        try Data("block log directory".utf8).write(to: support.appendingPathComponent("logs"))
        do {
            _ = try await fetcher.store.retainVerifiedDerived(
                model: model.id, revision: model.weights.revision, sourcePrecision: "fp16", derivedPrecision: "int4", thresholdBytes: 1)
            Issue.record("source deleted without durable log")
        } catch {}
        let source = try await fetcher.store.snapshotURL(model: model.id, revision: model.weights.revision, precision: "fp16")
        expectTrue(FileManager.default.fileExists(atPath: source.path))
        let receipt = support.appendingPathComponent("weights/retention/\(model.id)/\(model.weights.revision)/fp16.json")
        expectTrue(FileManager.default.fileExists(atPath: receipt.path)) // Receipt is persisted before the failed log; removal has not happened.
        let notRemoved = try await fetcher.store.sourceRetentionReceipt(model: model.id, revision: model.weights.revision, sourcePrecision: "fp16"); expectEqual(notRemoved, nil)
    }
    @Test func testLocalHFCacheImportsCloneWithoutNetworkOrSourceMutation() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, model) = try fixture(root)
        let cache = root.appendingPathComponent("hub"); let blob = cache.appendingPathComponent("blobs/source")
        try FileManager.default.createDirectory(at: blob.deletingLastPathComponent(), withIntermediateDirectories: true); try Self.payload.write(to: blob)
        let candidate = cache.appendingPathComponent("models--test--good/snapshots/\(model.weights.revision)/weights.bin")
        try FileManager.default.createDirectory(at: candidate.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: candidate, withDestinationURL: blob)
        let before = try FileManager.default.attributesOfItem(atPath: blob.path)
        let events = ProgressCapture(); var offline = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:9")!); offline.maximumAttempts = 1
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), client: offline, localSources: [cache.path])
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", progress: { events.add($0) })
        expectEqual(try Data(contentsOf: result.weightsDirectory.appendingPathComponent("weights.bin")), Self.payload)
        let recorded = events.read(); expectTrue(recorded.contains { $0.phase == "verifyingLocalCopy" }); expectFalse(recorded.contains { $0.phase == "downloading" })
        expectTrue(recorded.allSatisfy { $0.contentID == file().sha256 })
        let after = try FileManager.default.attributesOfItem(atPath: blob.path)
        expectEqual(before[.modificationDate] as? Date, after[.modificationDate] as? Date); expectEqual(before[.posixPermissions] as? Int, after[.posixPermissions] as? Int)
        let owned = try FileManager.default.attributesOfItem(atPath: result.weightsDirectory.appendingPathComponent("weights.bin").path)
        expectNotEqual(before[.systemFileNumber] as? UInt64, owned[.systemFileNumber] as? UInt64)
        try Data("changed source".utf8).write(to: blob)
        expectEqual(try Data(contentsOf: result.weightsDirectory.appendingPathComponent("weights.bin")), Self.payload)
    }
    /// Another app once linked its own Drama 3 conversion into the Fish Audio snapshot path, and a download of Arsenal's
    /// Fish pins wrote through the link over that app's weights. A cache path that resolves into another app's folder is neither a
    /// donor nor a destination: Get downloads into its own store and leaves the other folder byte for byte as it was.
    @Test func testSnapshotLinkedIntoAnotherAppsFolderIsNeverUsedOrWritten() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, model) = try fixture(root)
        let foreign = root.appendingPathComponent("other-app/voice-model")
        try FileManager.default.createDirectory(at: foreign, withIntermediateDirectories: true)
        try Self.payload.write(to: foreign.appendingPathComponent("weights.bin"))
        let before = try FileManager.default.attributesOfItem(atPath: foreign.appendingPathComponent("weights.bin").path)
        let hub = root.appendingPathComponent("hub"); let snapshots = hub.appendingPathComponent("models--test--good/snapshots")
        try FileManager.default.createDirectory(at: snapshots, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: snapshots.appendingPathComponent(model.weights.revision), withDestinationURL: foreign)
        let sources = LocalModelSources(paths: [hub.path])
        expectEqual(try sources.candidates(repo: "test/good", revision: model.weights.revision, path: "weights.bin"), [])
        let events = ProgressCapture(); let support = root.appendingPathComponent("support")
        let fetcher = ModelFetcher(support: support, client: client(server), localSources: [hub.path])
        let plan = try await fetcher.availabilityPlan(model: model, precision: "fp16")
        expectEqual(plan.localCopyBytes, 0); expectEqual(plan.networkBytes, Int64(Self.payload.count))
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", progress: { events.add($0) })
        expectEqual(try Data(contentsOf: result.weightsDirectory.appendingPathComponent("weights.bin")), Self.payload)
        expectFalse(events.read().contains { $0.phase == "verifyingLocalCopy" })
        expectTrue(LocalModelSources.resolves(result.weightsDirectory.appendingPathComponent("weights.bin"), inside: support))
        expectFalse(LocalModelSources.resolves(result.weightsDirectory.appendingPathComponent("weights.bin"), inside: foreign))
        let after = try FileManager.default.attributesOfItem(atPath: foreign.appendingPathComponent("weights.bin").path)
        expectEqual(before[.systemFileNumber] as? UInt64, after[.systemFileNumber] as? UInt64)
        expectEqual(before[.modificationDate] as? Date, after[.modificationDate] as? Date)
        expectEqual(try FileManager.default.contentsOfDirectory(atPath: foreign.path), ["weights.bin"])
        // A real cache snapshot (file links into its own blob store) is still a donor.
        let blob = hub.appendingPathComponent("models--test--real/blobs/source")
        try FileManager.default.createDirectory(at: blob.deletingLastPathComponent(), withIntermediateDirectories: true); try Self.payload.write(to: blob)
        let real = hub.appendingPathComponent("models--test--real/snapshots/\(model.weights.revision)/weights.bin")
        try FileManager.default.createDirectory(at: real.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: real, withDestinationURL: blob)
        expectEqual(try sources.candidates(repo: "test/real", revision: model.weights.revision, path: "weights.bin").count, 1)
    }
    @Test func testLocalHashMismatchDownloadsOnlyBadFileAndKeepsSource() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, model) = try fixture(root)
        let local = root.appendingPathComponent("local/test/good/weights.bin")
        try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
        let wrong = Data(repeating: 120, count: Self.payload.count); try wrong.write(to: local)
        let events = ProgressCapture()
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), client: client(server), localSources: [root.appendingPathComponent("local").path])
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", progress: { events.add($0) })
        expectEqual(try Data(contentsOf: result.weightsDirectory.appendingPathComponent("weights.bin")), Self.payload)
        expectEqual(try Data(contentsOf: local), wrong)
        let eventsRead = events.read(); expectTrue(eventsRead.contains { $0.phase == "verifyingLocalCopy" })
        expectTrue(eventsRead.contains { $0.phase == "downloading" && ($0.networkBytes ?? 0) > 0 })
        expectTrue(eventsRead.allSatisfy { $0.contentID == file().sha256 })
    }
    @Test func testPartialLocalSourceDownloadsMissingFilesOnly() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, initial) = try fixture(root); var model = initial
        let config = Data(String(repeating: "arsenal-config\n", count: 32).utf8)
        model.weights.files.append("config.json"); model.weights.sha256?["config.json"] = FileDigest.sha256(config); model.weights.bytes?["config.json"] = Int64(config.count)
        let source = root.appendingPathComponent("omlx/test/good/weights.bin")
        try FileManager.default.createDirectory(at: source.deletingLastPathComponent(), withIntermediateDirectories: true); try Self.payload.write(to: source)
        let events = ProgressCapture()
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), client: client(server), localSources: [root.appendingPathComponent("omlx").path])
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16", progress: { events.add($0) })
        expectEqual(try Data(contentsOf: result.weightsDirectory.appendingPathComponent("config.json")), config)
        let downloads = events.read().filter { $0.phase == "downloading" }; expectTrue(!downloads.isEmpty)
        expectTrue(downloads.allSatisfy { $0.file == "config.json" && $0.contentID == FileDigest.sha256(config) })
        expectEqual(try FileDigest.sha256(source), file().sha256)
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_REAL_LOCAL"] == "1")) func testRealQwenLocalImportZeroNetwork() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = repository.appendingPathComponent("models/qwen3.8-27b"); let model = try ModelManifest.load(folder)
        guard let path = ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"], path.contains("/Arsenal-dev/") else {
            throw ManifestError.invalid("real local import requires an explicit Arsenal-dev support directory")
        }
        let support = URL(fileURLWithPath: path)
        let source = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".omlx/models/Qwen/Qwen3.8-27B")
        var before: [String: Date] = [:]
        for file in model.weights.files { before[file] = try source.appendingPathComponent(file).resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate }
        var offline = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:9")!); offline.maximumAttempts = 1
        let events = ProgressCapture()
        let fetcher = ModelFetcher(support: support, client: offline, localSources: [FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".omlx/models").path])
        let beforePlan = try await fetcher.availabilityPlan(model: model, precision: "bf16")
        expectEqual(beforePlan.networkBytes, 0); expectEqual(beforePlan.localCopyBytes, model.downloadBytes(precision: "bf16"))
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "bf16", progress: { events.add($0) })
        _ = try await fetcher.verifyInstalled(model: model, precision: "bf16") // Recorded large blobs are stat-checked, not rehashed at Load.
        let afterPlan = try await fetcher.availabilityPlan(model: model, precision: "bf16")
        expectEqual(afterPlan.networkBytes, 0); expectEqual(afterPlan.localCopyBytes, 0); expectEqual(afterPlan.presentBytes, model.downloadBytes(precision: "bf16"))
        let snapshot = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: result.weightsDirectory.appendingPathComponent("arsenal-snapshot.json")))
        expectEqual(Set(snapshot.files.map(\.path)), Set(model.weights.files))
        let recorded = events.read(); expectFalse(recorded.contains { $0.phase == "downloading" }); expectEqual(recorded.compactMap(\.networkBytes).reduce(0, +), 0)
        let byHash = Dictionary(grouping: recorded, by: { $0.contentID ?? "" }).mapValues { $0.map(\.completedBytes).max() ?? 0 }
        expectEqual(Set(byHash.keys), Set(model.weights.sha256!.values))
        let bytes = byHash.values.reduce(0, +); expectEqual(bytes, model.downloadBytes(precision: "bf16"))
        for file in model.weights.files {
            expectEqual(try source.appendingPathComponent(file).resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate, before[file])
        }
        print("Qwen local Get verified \(snapshot.files.count) files, bytes=\(bytes), network_bytes=0, support=\(support.path), sources_unchanged=true")
    }
    @Test func testAcceptedPinsRejectSnapshotSelfHashTamper() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, initial) = try fixture(root); var model = initial; model.derive = ["int4": "tiny"]
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), client: client(server), converter: TestConverter(), localSources: [])
        let result = try await fetcher.get(model: model, modelDirectory: folder, precision: "fp16")
        _ = try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "fp16")
        let bad = Data(repeating: 33, count: Self.payload.count)
        let weights = result.weightsDirectory.appendingPathComponent("weights.bin")
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: weights.path); try bad.write(to: weights)
        let path = result.weightsDirectory.appendingPathComponent("arsenal-snapshot.json")
        var changed = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: path)); changed.files[0].sha256 = FileDigest.sha256(bad)
        try JSONEncoder().encode(changed).write(to: path)
        // Its forged self-hash passes the raw integrity reader; accepted catalog pins must still reject it.
        _ = try await fetcher.store.verifySnapshot(model: model.id, revision: model.weights.revision, precision: "fp16")
        do { _ = try await fetcher.store.verifyInstalledSnapshot(model: model, precision: "fp16"); Issue.record("forged self-hash accepted as catalog weights") } catch {}
    }
    @Test func testAvailabilityPlanIsReadOnlyAndSeparatesLocalNetworkPresent() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root); var model = initial
        let config = Data(String(repeating: "arsenal-config\n", count: 32).utf8)
        model.weights.files.append("config.json"); model.weights.sha256?["config.json"] = FileDigest.sha256(config); model.weights.bytes?["config.json"] = Int64(config.count)
        let candidate = root.appendingPathComponent("sources/test/good/weights.bin")
        try FileManager.default.createDirectory(at: candidate.deletingLastPathComponent(), withIntermediateDirectories: true); try Self.payload.write(to: candidate)
        let support = root.appendingPathComponent("support"); let fetcher = ModelFetcher(support: support, localSources: [root.appendingPathComponent("sources").path])
        let first = try await fetcher.availabilityPlan(model: model, precision: "fp16")
        expectEqual(first.localCopyBytes, file().bytes); expectEqual(first.networkBytes, Int64(config.count)); expectEqual(first.presentBytes, 0)
        expectFalse(FileManager.default.fileExists(atPath: support.path))
        let part = root.appendingPathComponent("part"); try Self.payload.write(to: part); try await fetcher.store.promote(part, sha256: file().sha256, bytes: file().bytes)
        let second = try await fetcher.availabilityPlan(model: model, precision: "fp16")
        expectEqual(second.localCopyBytes, 0); expectEqual(second.presentBytes, file().bytes)
        _ = folder
    }
    @Test func testLargeBlobUsesRecordedIdentityUntilMetadataChanges() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root); let data = Data(repeating: 7, count: 17 * 1024 * 1024)
        let file = DerivedFile(path: "model.safetensors", bytes: Int64(data.count), sha256: FileDigest.sha256(data))
        let part = root.appendingPathComponent("part"); try data.write(to: part); try await store.promote(part, sha256: file.sha256, bytes: file.bytes)
        let revision = String(repeating: "a", count: 40)
        let directory = try await store.commitSnapshot(SnapshotManifest(model: "large-test", revision: revision, precision: "bf16", files: [file], derived: nil))
        let blob = try await store.blobURL(sha256: file.sha256)
        // Same inode/mtime/size: a verification read must not open/hash the unreadable payload.
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: blob.path)
        _ = try await store.verifySnapshot(model: "large-test", revision: revision, precision: "bf16")
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: blob.path)
        let handle = try FileHandle(forWritingTo: blob); try handle.write(contentsOf: Data([9])); try handle.close()
        let present = try await store.isVerifiedPresent(sha256: file.sha256, bytes: file.bytes); expectFalse(present)
        do { _ = try await store.verifySnapshot(model: "large-test", revision: revision, precision: "bf16"); Issue.record("changed payload skipped re-hash") } catch {}
        expectTrue(FileManager.default.fileExists(atPath: directory.path))
    }
    @Test func testIncrementalDownloadsUseStallNotTotalDeadline() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        var c = client(server); c.stallTimeout = 0.2; c.maximumAttempts = 1
        let path = try await c.download(repo: "test/slow", revision: String(repeating: "a", count: 40), file: file(), directory: root.appendingPathComponent("slow"))
        expectEqual(try Data(contentsOf: path), Self.payload)
        c.stallTimeout = 0.05
        do {
            _ = try await c.download(repo: "test/stalled", revision: String(repeating: "a", count: 40), file: file(), directory: root.appendingPathComponent("stalled"))
            Issue.record("stall was not bounded")
        } catch { expectTrue(error.localizedDescription.contains("stall") || (error as NSError).code == NSURLErrorTimedOut) }
    }
    @Test func testRollbackChoosesHighestVerifiedCatalog() throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let key = Curve25519.Signing.PrivateKey(); let (_, model) = try fixture(root)
        let entry = CatalogModel(manifest: model, files: [.init(path: "model.toml", sha256: String(repeating: "a", count: 64), bytes: 1)])
        let bundle = root.appendingPathComponent("bundle"); let cached = root.appendingPathComponent("cached")
        for (directory, sequence) in [(bundle, Int64(2)), (cached, Int64(3))] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let data = try CatalogDocument(sequence: sequence, sourceCommit: String(repeating: "a", count: 40), models: [entry]).encoded()
            try data.write(to: directory.appendingPathComponent("catalog.json")); try key.signature(for: data).write(to: directory.appendingPathComponent("catalog.json.sig"))
        }
        let chosen = try CatalogBootstrap.select(bundledDirectory: bundle, cachedDirectory: cached, publicKey: key.publicKey.rawRepresentation)
        expectEqual(chosen.document.sequence, 3); expectEqual(chosen.document.models.count, 1)
        try Data([1]).write(to: bundle.appendingPathComponent("catalog.json.sig"))
        expectEqual(try CatalogBootstrap.select(bundledDirectory: bundle, cachedDirectory: cached, publicKey: key.publicKey.rawRepresentation).document.sequence, 3)
    }
    @Test func testDerivedOutputReservationsSurviveUnrelatedDelete() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let store = BlobStore(support: root)
        let generated = Data("generated weights".utf8); let f = DerivedFile(path: "model.safetensors", bytes: Int64(generated.count), sha256: FileDigest.sha256(generated))
        let part = root.appendingPathComponent("derived-output"); try generated.write(to: part)
        await store.reserve(files: [f]); try await store.promote(part, sha256: f.sha256, bytes: f.bytes)
        try await store.delete(model: "unrelated")
        _ = try await store.commitAlreadyVerifiedSnapshot(
            SnapshotManifest(model: "derived", revision: String(repeating: "a", count: 40), precision: "int4", files: [f], derived: nil))
        await store.unreserve(files: [f]); try await store.delete(model: "unrelated")
        let present = try await store.hasBlob(sha256: f.sha256, bytes: f.bytes); expectTrue(present)
    }
    @Test func testFreshBundleLayoutReadsSourceAndGetsSignedFolder() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let resources = root.appendingPathComponent("bundle-fixture/Contents/Resources"); let cat = resources.appendingPathComponent("catalog")
        try FileManager.default.createDirectory(at: cat, withIntermediateDirectories: true)
        for name in ["catalog.json", "catalog.json.sig"] {
            try FileManager.default.copyItem(at: repository.appendingPathComponent("Resources/catalog/" + name), to: cat.appendingPathComponent(name))
        }
        try FileManager.default.copyItem(at: repository.appendingPathComponent("Resources/catalog-public-key"), to: resources.appendingPathComponent("catalog-public-key"))
        try JSONSerialization.data(withJSONObject: ["local_git": LocalGitCatalogSource.defaultRepository.path]).write(to: cat.appendingPathComponent("source.json"))
        // The signed catalog names the commit it was built from. A checkout without that commit (the public repository's
        // fresh history, a shallow clone) cannot serve folders from it; the shipped app reads its bundle instead.
        guard catalogSourceCommitIsPresent(repository: repository, catalog: cat.appendingPathComponent("catalog.json")) else { return }
        let support = root.appendingPathComponent("fresh-support"); let key = try Data(contentsOf: resources.appendingPathComponent("catalog-public-key"))
        let selection = try CatalogBootstrap.select(bundledDirectory: cat, cachedDirectory: support.appendingPathComponent("catalog"), publicKey: key)
        expectTrue(Set(["flux2-klein-4b", "kokoro-82m", "laya-multilingual"]).isSubset(of: Set(selection.document.models.map(\.id))))
        let source = try CatalogBootstrap.bundledSource(resources: resources)
        let installer = ModelFolderInstaller(support: support); try await installer.accept(data: selection.data, signature: selection.signature, publicKey: key)
        let folder = try await installer.install(id: "kokoro-82m", source: source)
        expectEqual(try ModelManifest.load(folder).id, "kokoro-82m")
        let verified = try await installer.verifyInstalled(id: "kokoro-82m"); expectEqual(verified.path, folder.path)
    }
    @Test func testNativeContextIsOptionalAndSeparateFromConfiguredContext() throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (_, missing) = try fixture(root); expectEqual(missing.nativeContext, nil)
        let (folder, model) = try fixture(root, extras: "context = 32768\nnative_context = 262144")
        expectEqual(model.context, 32768); expectEqual(model.nativeContext, 262144)
        let encoded = try JSONEncoder().encode(model); expectTrue(String(data: encoded, encoding: .utf8)!.contains("native_context"))
        var invalid = model; invalid.nativeContext = 0; expectFalse(invalid.problems(folder: folder).isEmpty)
    }
    @Test func testRunnerChoiceUsesOnlyAcceptedFolderDefaultAndWritesNoCache() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let (_, original) = try fixture(root)
        let selector = RunnerSelector(support: root); var model = original
        model.selection = RunnerSelection(defaultRunner: "native", chips: ["m4": "native"])
        let env = RunnerSelectionEnvironment(chip: "m4-max", macOS: "26.6", installedWorkers: ["chat"])
        let choice = try await selector.select(model: model, environment: env, fast: true); expectEqual(choice.runner, "native")
        expectFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("state/runner-choice.json").path))
        model.selection = RunnerSelection(defaultRunner: "missing")
        do { _ = try await selector.select(model: model, environment: env, fast: false); Issue.record("missing accepted runner fell back") } catch {}
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_REAL_HF"] == "1")) func testRealHuggingFaceKokoro() async throws {
        let root = try temp(); let c = HuggingFaceClient()
        let weights = WeightSpec(
            repo: "mlx-community/Kokoro-82M-bf16", revision: "a71e4d38b236d968966a2002c4c895dbd12b1c3c", dtype: "fp32",
            files: ["config.json", "kokoro-v1_0.safetensors", "voices/af_heart.safetensors"], sha256: nil, bytes: nil)
        do {
            let files = try await c.pins(weights)
            for f in files {
                let downloaded = try await c.download(repo: weights.repo, revision: weights.revision, file: f, directory: root.appendingPathComponent("weights"))
                try FileDigest.verify(downloaded, sha256: f.sha256, bytes: f.bytes); print("HF verified \(f.path) bytes=\(f.bytes) sha256=\(f.sha256)")
            }
            try FileManager.default.removeItem(at: root)
        } catch {
            let preserved = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support/Arsenal-dev/weights/fetch/\(root.lastPathComponent)")
            try FileManager.default.createDirectory(at: preserved.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: root, to: preserved); throw error
        }
    }
}
/// TestConverter's transform, recording each call's recipe and source bytes.
private final class RecordingConverter: PrecisionConverter, @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [(recipe: String, source: Data)] = []
    var calls: [(recipe: String, source: Data)] { lock.withLock { recorded } }
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile] {
        let source = try Data(contentsOf: sourceDirectory.appendingPathComponent("weights.bin"))
        lock.withLock { recorded.append((recipe, source)) }
        let data = source + Data(precision.utf8)
        try data.write(to: destinationDirectory.appendingPathComponent("weights.bin"))
        return [DerivedFile(path: "weights.bin", bytes: Int64(data.count), sha256: FileDigest.sha256(data))]
    }
}
/// TestConverter's transform for the primary file, plus the `t5` asset stored under `text/` by the base conversion.
extension URL {
    fileprivate func createParent() throws -> URL {
        try FileManager.default.createDirectory(at: deletingLastPathComponent(), withIntermediateDirectories: true); return self
    }
}
private struct ComponentConverter: PrecisionConverter {
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile] {
        try await convert(
            model: model, modelDirectory: modelDirectory, sourceDirectory: sourceDirectory, assets: [:], destinationDirectory: destinationDirectory, precision: precision,
            recipe: recipe)
    }
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, assets: [String: URL], destinationDirectory: URL, precision: String, recipe: String)
        async throws -> [DerivedFile]
    {
        var outputs: [(String, Data)] = [("weights.bin", try Data(contentsOf: sourceDirectory.appendingPathComponent("weights.bin")) + Data(precision.utf8))]
        let stored = sourceDirectory.appendingPathComponent("text/model.bin")
        if FileManager.default.fileExists(atPath: stored.path) {
            outputs.append(("text/model.bin", try Data(contentsOf: stored) + Data(precision.utf8)))
        } else if let asset = assets["t5"], recipe.hasPrefix("cast:") {
            outputs.append(("text/model.bin", try Data(contentsOf: asset.appendingPathComponent("model.bin")) + Data(precision.utf8)))
        }
        return try outputs.map { path, data in
            let target = destinationDirectory.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: target)
            return DerivedFile(path: path, bytes: Int64(data.count), sha256: FileDigest.sha256(data))
        }
    }
}
private struct TestConverter: PrecisionConverter {
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile] {
        let data = try Data(contentsOf: sourceDirectory.appendingPathComponent("weights.bin")) + Data(precision.utf8)
        try data.write(to: destinationDirectory.appendingPathComponent("weights.bin"))
        return [DerivedFile(path: "weights.bin", bytes: Int64(data.count), sha256: FileDigest.sha256(data))]
    }
}
/// The loopback Hub stand-in (Fixtures/hf_server.py); shared with GatedFetchTests.
final class StandIn {
    private let process: ChildProcess
    private let lock = NSLock()
    private var stopped = false
    let url: URL
    init(root: URL) async throws {
        let server = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/hf_server.py")
        process = try ChildProcess(
            executable: URL(fileURLWithPath: "/usr/bin/python3"), arguments: ["-u", server.path, root.path], environment: RunnerEnvironment.scrub(),
            log: root.appendingPathComponent("server.log"))
        var line = Data()
        while let b = try await process.readOutputAsync(count: 1), !b.isEmpty, b != Data([10]) { line.append(b) }
        guard let port = String(data: line, encoding: .utf8), Int(port) != nil else {
            process.signal(SIGKILL); _ = process.awaitExit(); throw ManifestError.invalid("test server did not start")
        }
        url = URL(string: "http://127.0.0.1:\(port)/")!
    }
    func stop() {
        guard
            lock.withLock({
                if stopped { return false }; stopped = true; return true
            })
        else { return }
        // Foundation waitUntilExit can wait on its run loop after the child has exited. Fixtures use the same
        // exact-PID, synchronous reaper as runners instead; they have no graceful-shutdown state to preserve.
        process.signal(SIGKILL); _ = process.awaitExit(); process.closeInput(); process.finishOutput()
    }
}

private func expectEqual<T: Equatable>(_ a: T, _ b: T) { #expect(a == b) }
private func expectNotEqual<T: Equatable>(_ a: T, _ b: T) { #expect(a != b) }
private func expectTrue(_ value: Bool) { #expect(value) }
private func catalogSourceCommitIsPresent(repository: URL, catalog: URL) -> Bool {
    guard let data = try? Data(contentsOf: catalog) else { return false }
    guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
    guard let commit = object["source_commit"] as? String else { return false }
    let git = Process(); git.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    git.arguments = ["-C", repository.path, "cat-file", "-e", commit + "^{commit}"]
    git.standardOutput = FileHandle.nullDevice; git.standardError = FileHandle.nullDevice
    guard (try? git.run()) != nil else { return false }
    git.waitUntilExit()
    return git.terminationStatus == 0
}
private func expectFalse(_ value: Bool) { #expect(!value) }
private func expectThrows<T>(_ value: @autoclosure () throws -> T) { #expect(throws: (any Error).self) { _ = try value() } }

private final class ProgressCapture: @unchecked Sendable {
    private let lock = NSLock(); private var events: [DownloadProgress] = []
    func add(_ event: DownloadProgress) { lock.withLock { events.append(event) } }
    func read() -> [DownloadProgress] { lock.withLock { events } }
}

private actor InterruptingCastConverter: PrecisionConverter {
    var calls = 0
    let cancel: Bool
    init(cancel: Bool) { self.cancel = cancel }
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile] {
        calls += 1
        if calls == 2 {
            if cancel { withUnsafeCurrentTask { $0?.cancel() }; try Task.checkCancellation() }
            throw ManifestError.invalid("Injected second-file failure")
        }
        let name = try FileManager.default.contentsOfDirectory(atPath: sourceDirectory.path).first!
        let data = try Data(contentsOf: sourceDirectory.appendingPathComponent(name)) + Data(precision.utf8)
        try data.write(to: destinationDirectory.appendingPathComponent(name))
        return [DerivedFile(path: name, bytes: Int64(data.count), sha256: FileDigest.sha256(data))]
    }
}

private actor NativeFileConverter: PrecisionConverter {
    var calls: [Int] = []
    func counts() -> [Int] { calls }
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile] {
        let paths = try SafePath.files(in: sourceDirectory)
        calls.append(paths.count)
        return try paths.map { path in
            let output = destinationDirectory.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            let bytes = try Data(contentsOf: sourceDirectory.appendingPathComponent(path)) + Data(precision.utf8)
            try bytes.write(to: output)
            return DerivedFile(path: path, bytes: Int64(bytes.count), sha256: FileDigest.sha256(bytes))
        }
    }
}

extension FetchTests {
    @Test(arguments: ["fp16", "bf16"]) func nativeNormalizationIsGenericAndDropsEachOriginal(servingHalf: String) async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (folder, initial) = try fixture(root, dtype: "bf16", worker: "audio")
        var model = initial
        model.precision = NativeHalfPolicy(dtype: servingHalf, reason: "Synthetic documented exception", evidence: "log/001-test.md")
        model.derive = ["int8": "tiny", "int4": "tiny"]
        model.weights.files = ["nested/a.bin", "tokenizer/b.bin"]
        model.weights.sha256 = Dictionary(uniqueKeysWithValues: model.weights.files.map { ($0, FileDigest.sha256(Self.payload)) })
        model.weights.bytes = Dictionary(uniqueKeysWithValues: model.weights.files.map { ($0, Int64(Self.payload.count)) })
        for name in model.weights.files {
            let donor = root.appendingPathComponent("local/test/good/" + name)
            try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true); try Self.payload.write(to: donor)
        }
        let converter = NativeFileConverter()
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), converter: converter, localSources: [root.appendingPathComponent("local").path])
        if servingHalf == "fp16" {
            // Upgrade an old Get that committed the original BF16 as a serving snapshot.
            let hash = FileDigest.sha256(Self.payload)
            _ = try await fetcher.store.importLocal(file: root.appendingPathComponent("local/test/good/nested/a.bin"), sha256: hash, bytes: Int64(Self.payload.count))
            _ = try await fetcher.store.commitSnapshot(
                SnapshotManifest(
                    model: model.id, revision: model.weights.revision, precision: "bf16",
                    files: model.weights.files.map { DerivedFile(path: $0, bytes: Int64(Self.payload.count), sha256: hash) }, derived: nil))
        }
        #expect(model.basePrecision == servingHalf && model.source(for: servingHalf) == nil)
        _ = try await fetcher.get(model: model, modelDirectory: folder, precision: servingHalf, approvedDownloads: [:])
        let snapshot = try #require(try await fetcher.store.verifyInstalledSnapshot(model: model, precision: servingHalf, forceHash: true))
        #expect(snapshot.derived?.normalization == "native-half-v1:" + servingHalf)
        #expect(snapshot.derived?.recipe == NativeHalfPolicy.recipe)
        #expect(await converter.counts() == [1, 1])
        if servingHalf == "fp16" {
            let old = try await fetcher.store.snapshotURL(model: model.id, revision: model.weights.revision, precision: "bf16")
            #expect(!FileManager.default.fileExists(atPath: old.path))
        }
        #expect(try await fetcher.store.hasBlob(sha256: FileDigest.sha256(Self.payload), bytes: Int64(Self.payload.count)) == false)
        #expect(try await fetcher.updateRequired(model: model, precision: servingHalf) == false)
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("support/weights").path).allSatisfy { !$0.hasPrefix(".cast-") })
    }
}

/// Converts every model file in the source view (a snapshot's own manifests are the store's, never outputs).
private struct ShardConverter: PrecisionConverter {
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile] {
        try SafePath.files(in: sourceDirectory).filter { !["arsenal-snapshot.json", "arsenal-derived.json"].contains($0) }.map { path in
            let output = destinationDirectory.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            let bytes = try Data(contentsOf: sourceDirectory.appendingPathComponent(path)) + Data(precision.utf8)
            try bytes.write(to: output)
            return DerivedFile(path: path, bytes: Int64(bytes.count), sha256: FileDigest.sha256(bytes))
        }
    }
}

private final class OrderedGetLog: @unchecked Sendable {
    private let lock = NSLock(); private var events: [DownloadProgress] = []
    func progress(_ event: DownloadProgress) { lock.withLock { events.append(event) } }
    func deriving() {
        lock.withLock {
            events.append(
                DownloadProgress(file: "", completedBytes: 0, totalBytes: 0, bytesPerSecond: 0, contentID: GetProgressDelivery.derivingContentID, phase: "deriving"))
        }
    }
    func read() -> [DownloadProgress] { lock.withLock { events } }
}

extension FetchTests {
    /// Live fault (20 Oct): the reranker's fp16 copy was removed by source retention; the next quantized Get
    /// re-downloaded the BF16 source shard by shard, converting each before fetching the next. The row froze on
    /// "deriving" after the first shard. Reproduced here with two small shards and the real fetch order.
    @Test func removedSourceRestoreFailsFastWithoutApprovalAndShowsEveryShardDownload() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }; let server = try await StandIn(root: root); defer { server.stop() }
        let (folder, initial) = try fixture(root, dtype: "bf16", worker: "audio")
        var model = initial
        model.precision = NativeHalfPolicy(dtype: "fp16")
        model.derive = ["int8": "tiny", "int4": "tiny"]
        let config = Data(String(repeating: "arsenal-config\n", count: 32).utf8)
        model.weights.files = ["config.json", "weights.bin"]
        model.weights.sha256 = ["config.json": FileDigest.sha256(config), "weights.bin": FileDigest.sha256(Self.payload)]
        model.weights.bytes = ["config.json": Int64(config.count), "weights.bin": Int64(Self.payload.count)]
        let shards = try model.weights.files.map { path in
            DerivedFile(path: path, bytes: try #require(model.weights.bytes?[path]), sha256: try #require(model.weights.sha256?[path]))
        }
        let fetcher = ModelFetcher(support: root.appendingPathComponent("support"), client: client(server), converter: ShardConverter())
        _ = try await fetcher.get(model: model, modelDirectory: folder, precision: "int8")
        let receipt = try await fetcher.store.retainVerifiedDerived(
            model: model.id, revision: model.weights.revision, sourcePrecision: "fp16", derivedPrecision: "int8", thresholdBytes: 1)
        #expect(receipt != nil)
        for shard in shards { #expect(try await fetcher.store.hasBlob(sha256: shard.sha256, bytes: shard.bytes) == false) }

        // A plan reviewed before the removal approved no download: refused at the first shard, nothing converted.
        let refused = OrderedGetLog()
        do {
            _ = try await fetcher.get(
                model: model, modelDirectory: folder, precision: "int4", approvedDownloads: [:], progress: { refused.progress($0) }, deriving: { refused.deriving() })
            Issue.record("Restoring a removed source ran without an approved download")
        } catch let failure as WorkerFailure { #expect(failure.code == "get_plan_changed") }
        #expect(!refused.read().contains { $0.contentID == GetProgressDelivery.derivingContentID })

        let log = OrderedGetLog()
        let approved = Dictionary(uniqueKeysWithValues: shards.map { ($0.sha256, $0.bytes) })
        _ = try await fetcher.get(
            model: model, modelDirectory: folder, precision: "int4", approvedDownloads: approved, progress: { log.progress($0) }, deriving: { log.deriving() })
        var progress = try GetProgressAccumulator(files: shards.map { PendingDownloadFile(contentID: $0.sha256, file: $0.path, bytes: $0.bytes) })
        let states = log.read().compactMap { progress.record($0) }
        let firstDeriving = try #require(states.firstIndex { $0.state == .deriving })
        let bytesAtDeriving = states[..<firstDeriving].last?.bytes ?? 0
        // The second shard downloads after the first conversion step: the row must say so and count its bytes.
        #expect(states[firstDeriving...].contains { $0.state == .downloading && ($0.bytes ?? 0) > bytesAtDeriving })
        #expect(states.last { $0.state == .downloading }?.bytes == shards.reduce(0) { $0 + $1.bytes })
        #expect(states.last?.state == .deriving)
    }
}
