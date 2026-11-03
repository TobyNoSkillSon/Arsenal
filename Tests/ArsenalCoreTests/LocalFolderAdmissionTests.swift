// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// A local-models folder is admitted only when it is finished: no placeholder left from the scaffold (the kit's
/// S1-todo rule), and every pinned weights file present in its Hugging Face repository at the pinned revision. Either
/// failure is an error row naming the problem, refused by approval and Get. No network: the repository listing is
/// injected, and the client tests use the loopback stand-in.
@Suite struct LocalFolderAdmissionTests {
    /// Counts listings, so a cached check is seen not to list again.
    actor Lister {
        var calls = 0
        var answer: PinnedFilesListing
        /// Per-repository answers, over `answer`.
        var byRepo: [String: PinnedFilesListing] = [:]
        init(_ answer: PinnedFilesListing, byRepo: [String: PinnedFilesListing] = [:]) { self.answer = answer; self.byRepo = byRepo }
        func list(_ repo: String) -> PinnedFilesListing { calls += 1; return byRepo[repo] ?? answer }
        func set(_ answer: PinnedFilesListing) { self.answer = answer }
        nonisolated var lister: PinnedFilesLister { { repo, _, _ in await self.list(repo) } }
    }
    /// The fixture's voices model copied as the local folder `local-voices`, with `edit` applied to its model.toml.
    func writeLocal(_ f: LocalCatalogFixture, edit: (String) -> String = { $0 }) throws {
        let local = f.paths.support.appendingPathComponent("local-models/local-voices")
        for (path, data) in f.source.files where path.hasPrefix("models/fixture-voices/") {
            let relative = String(path.dropFirst("models/fixture-voices/".count)), file = local.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            let text = String(decoding: data, as: UTF8.self).replacingOccurrences(of: "fixture-voices", with: "local-voices")
            try (relative == "model.toml" ? Data(edit(text).utf8) : data).write(to: file)
        }
    }
    func errorRow(_ f: LocalCatalogFixture) async throws -> ModelRow? {
        await f.service.catalog().map(\.row).first { $0.name == "local-voices" && $0.id.hasPrefix("local-error-") }
    }

    // MARK: Placeholders

    @Test func placeholderNamesTheTomlField() {
        let name = LocalFolderPlaceholders.placeholder(in: "schema = 3\nname = \"TODO: full official name\"\n", path: "model.toml")
        #expect(name == "model.toml name is still a placeholder (\"TODO: full official name…\"): replace it with the real value from the model card, then refresh.")
        let facet = LocalFolderPlaceholders.placeholder(in: "name = \"X\"\n[facets]\nmodalities_in = [\"TODO\"]\n", path: "model.toml")
        #expect(facet?.hasPrefix("model.toml [facets] modalities_in is still a placeholder") == true)
        let list = LocalFolderPlaceholders.placeholder(in: "limits = [\n  \"fine\",\n  \"TODO: where it fails\",\n]\n", path: "model.toml")
        #expect(list?.hasPrefix("model.toml limits is still a placeholder") == true)
        #expect(LocalFolderPlaceholders.placeholder(in: "# TODO: full name\n", path: "README.md")?.hasPrefix("README.md is still a placeholder") == true)
        // The kit's rule is the whole upper-case word: a longer word, lower case or a word containing it is real text.
        for text in ["TODOS = 1", "todo: lower case", "MASTODON"] { #expect(LocalFolderPlaceholders.placeholder(in: text, path: "model.toml") == nil) }
    }

    @Test func placeholderOnlyInCheckedFilesAndModelTomlFirst() throws {
        let folder = URL(fileURLWithPath: "/private/tmp/arsenal-placeholders-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: folder) }
        let files = [
            "README.md": "# TODO: name", "model.toml": "summary = \"TODO: one sentence\"", "weights.bin": "TODO", "runners/swift/bin/tool.json": "TODO",
            "runners/swift/.build/x.swift": "let note = \"TODO\"", "ops/run.input.json": "{}"
        ]
        for (path, text) in files {
            let url = folder.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true); try Data(text.utf8).write(to: url)
        }
        let all = files.map { (path: $0.key, bytes: Int64($0.value.utf8.count)) }
        #expect(LocalFolderPlaceholders.first(in: folder, files: all)?.hasPrefix("model.toml summary") == true)
        let rest = all.filter { $0.path != "model.toml" && $0.path != "README.md" }
        #expect(LocalFolderPlaceholders.first(in: folder, files: rest) == nil)
        #expect(LocalFolderPlaceholders.first(in: folder, files: [(path: "README.md", bytes: LocalFolderPlaceholders.maxBytes + 1)]) == nil)
    }

    @Test func placeholderFolderIsAnErrorRowNeverApprovable() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        try writeLocal(f) { $0.replacingOccurrences(of: "name = \"Fixture voices\"", with: "name = \"TODO: full official name with family, size and version\"") }
        #expect(!(await f.service.catalog().contains { $0.id == "local-voices" }))
        let row = try #require(try await errorRow(f))
        #expect(row.summary.hasPrefix("model.toml name is still a placeholder (\"TODO: full official name with family, size and version") == true)
        #expect(row.contentDigest == nil && !row.needsCodeConsent && row.states["bf16"]?.state == .failed)
        do {
            try await f.service.approveLocalCode(id: "local-voices", contentDigest: String(repeating: "0", count: 64), approvedBy: .mcp)
            Issue.record("A placeholder folder was approved")
        } catch let error as WorkerFailure {
            #expect(error.code == "local_folder" && error.message.contains("model.toml name is still a placeholder"))
        }
        // Filled in: admitted as usual.
        try writeLocal(f)
        #expect(await f.service.catalog().contains { $0.id == "local-voices" })
        #expect(try await errorRow(f) == nil)
        await f.service.shutdown()
    }

    // MARK: Pinned files

    @Test func checkComparesEveryPinnedSource() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let model = try await f.service.manifest("fixture-voices")
        #expect(await PinnedFilesCheck.run(model, lister: Lister(.files(["base.bin", "README.md"])).lister) == .present)
        let missing = await PinnedFilesCheck.run(model, lister: Lister(.files(["README.md"])).lister)
        #expect(missing == .missing(repo: "test/fixture", revision: f.revision, files: ["base.bin"]))
        #expect(missing.problem?.hasPrefix("test/fixture has no file base.bin at revision aaaaaaaaaaaa: model.toml pins a file Hugging Face does not have") == true)
        #expect(await PinnedFilesCheck.run(model, lister: Lister(.repoMissing(hasToken: true)).lister).problem?.hasPrefix("Hugging Face has no repository test/fixture") == true)
        let gated = await PinnedFilesCheck.run(model, lister: Lister(.gated(repo: "test/fixture")).lister)
        #expect(gated.problem == nil && gated.note?.contains("gated") == true)
        let offline = await PinnedFilesCheck.run(model, lister: Lister(.unreachable("offline")).lister)
        #expect(offline.problem == nil && !offline.isFinal && offline.note?.hasPrefix("Arsenal could not reach Hugging Face") == true)
        await f.service.shutdown()
    }

    @Test func missingPinIsAnErrorRowRefusedByApprovalAndGet() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        try writeLocal(f)
        let entry = try #require(await f.service.catalog().first { $0.id == "local-voices" })
        let digest = try #require(entry.row.contentDigest)
        try await f.service.add(id: "local-voices")
        // The repository at the pinned revision has no base.bin.
        let lister = Lister(.files(["config.json"]))
        await f.service.setPinnedFilesLister(lister.lister)
        for operation in ["approve", "get"] {
            do {
                if operation == "approve" {
                    try await f.service.approveLocalCode(id: "local-voices", contentDigest: digest, approvedBy: .mcp)
                } else {
                    try await f.service.get(id: "local-voices", precision: "bf16")
                }
                Issue.record("\(operation) admitted a folder pinning a file the repository lacks")
            } catch let error as WorkerFailure {
                #expect(error.code == "local_folder", "\(operation)")
                #expect(error.message.hasPrefix("test/fixture has no file base.bin at revision aaaaaaaaaaaa"), "\(operation): \(error.message)")
            }
        }
        let row = try #require(await f.service.catalog().first { $0.id == "local-voices" }?.row)
        #expect(row.states[row.selectedPrecision]?.state == .failed)
        #expect(row.states[row.selectedPrecision]?.error?.hasPrefix("test/fixture has no file base.bin at revision aaaaaaaaaaaa") == true)
        #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).localCodeApprovals["local-voices"] == nil)
        // One listing for this digest, however often discovery, approval and Get ask.
        #expect(await lister.calls == 1)
        await f.service.shutdown()
    }

    @Test func missingPinOnANewFolderIsAnErrorRowFromDiscovery() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        await f.service.setPinnedFilesLister(Lister(.files([])).lister)
        try writeLocal(f)
        _ = await f.service.catalog() // discovery admits the folder and starts the check in the background
        // Once listed, the folder seen before keeps its id and becomes an error row with the reason.
        func reason() async -> String? {
            await f.service.catalog().first { $0.id == "local-voices" }.flatMap { $0.row.states[$0.row.selectedPrecision]?.error }
        }
        var error: String?
        for _ in 0..<300 where error == nil { error = await reason(); if error == nil { try await Task.sleep(for: .milliseconds(10)) } }
        #expect(error?.hasPrefix("test/fixture has no file base.bin at revision aaaaaaaaaaaa") == true)
        #expect(await f.service.catalog().first { $0.id == "local-voices" }?.row.verification.hasPrefix("test/fixture has no file") == true)
        do {
            try await f.service.approveLocalCode(id: "local-voices", contentDigest: String(repeating: "0", count: 64), approvedBy: .mcp)
            Issue.record("approved")
        } catch let error as WorkerFailure { #expect(error.code == "local_folder") }
        await f.service.shutdown()
    }

    @Test(arguments: [PinnedFilesListing.unreachable("offline"), .gated(repo: "test/fixture"), .files(["base.bin"])])
    func uncheckablePinsLetApprovalProceedAndSaySo(answer: PinnedFilesListing) async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let lister = Lister(answer)
        await f.service.setPinnedFilesLister(lister.lister)
        try writeLocal(f)
        let entry = try #require(await f.service.catalog().first { $0.id == "local-voices" })
        try await f.service.approveLocalCode(id: "local-voices", contentDigest: try #require(entry.row.contentDigest), approvedBy: .mcp)
        #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).localCodeApprovals["local-voices"] != nil)
        let note = await f.service.pinnedFilesNote(id: "local-voices")
        switch answer {
        case .unreachable: #expect(note?.hasPrefix("Arsenal could not reach Hugging Face to check that test/fixture has the pinned files (offline)") == true)
        case .gated: #expect(note?.contains("is gated on Hugging Face") == true)
        default: #expect(note == nil)
        }
        #expect(try await errorRow(f) == nil)
        await f.service.shutdown()
    }

    @Test func noListerMeansNoCheck() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        try writeLocal(f)
        let entry = try #require(await f.service.catalog().first { $0.id == "local-voices" })
        try await f.service.approveLocalCode(id: "local-voices", contentDigest: try #require(entry.row.contentDigest), approvedBy: .mcp)
        #expect(await f.service.pinnedFilesNote(id: "local-voices") == nil)
        await f.service.shutdown()
    }

    // MARK: Tokens and scopes (review fixes)

    @Test func aRepositoryMissingWithoutAWorkingTokenIsRetriedAndSaysWhy() {
        let anonymous = PinnedFilesCheck.repoMissing(repo: "o/m", revision: String(repeating: "a", count: 40), hasToken: false)
        #expect(!anonymous.isFinal && anonymous.problem?.contains("Arsenal has no Hugging Face token") == true)
        let rejected = PinnedFilesCheck.repoMissing(repo: "o/m", revision: String(repeating: "a", count: 40), hasToken: false, tokenRejected: true)
        #expect(!rejected.isFinal && rejected.problem?.contains("refused Arsenal's token (expired or revoked") == true)
        #expect(rejected.problem?.contains("no Hugging Face token") == false)
        #expect(PinnedFilesCheck.repoMissing(repo: "o/m", revision: String(repeating: "a", count: 40), hasToken: true).isFinal)
    }

    @Test func savingATokenListsARepositoryThatLookedMissingAgain() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        try writeLocal(f)
        let entry = try #require(await f.service.catalog().first { $0.id == "local-voices" })
        let digest = try #require(entry.row.contentDigest)
        try await f.service.add(id: "local-voices")
        let lister = Lister(.repoMissing(hasToken: false))
        await f.service.setPinnedFilesLister(lister.lister)
        await #expect(throws: WorkerFailure.self) { try await f.service.approveLocalCode(id: "local-voices", contentDigest: digest, approvedBy: .mcp) }
        let failed = try #require(await f.service.catalog().first { $0.id == "local-voices" }?.row)
        #expect(failed.states[failed.selectedPrecision]?.error?.hasPrefix("Hugging Face has no repository test/fixture") == true)
        // The user saves a token in Settings; the private repository now lists.
        await lister.set(.files(["base.bin"]))
        await f.service.retryGatedGets()
        let cleared = try #require(await f.service.catalog().first { $0.id == "local-voices" }?.row)
        #expect(cleared.states[cleared.selectedPrecision]?.phase?.hasPrefix("localFolderError") != true)
        try await f.service.approveLocalCode(id: "local-voices", contentDigest: digest, approvedBy: .mcp)
        await f.service.shutdown()
    }

    /// The fixture's voices folder with its int4 tier published by the makers (`[vendor.int4]` in another repository).
    func writeVendorLocal(_ f: LocalCatalogFixture) throws {
        let hash = String(repeating: "d", count: 64)
        try writeLocal(f) { text in
            var lines = text.components(separatedBy: "\n").filter { !$0.hasPrefix("int4 = ") }
            lines += [
                "[vendor.int4]", "repo = \"test/vendor\"", "revision = \"\(f.revision)\"", "dtype = \"int4\"", "files = [\"v.bin\"]",
                "[vendor.int4.sha256]", "\"v.bin\" = \"\(hash)\"", "[vendor.int4.bytes]", "\"v.bin\" = 3"
            ]
            return lines.joined(separator: "\n") + "\n"
        }
    }

    @Test func aBadVendorPinBlocksOnlyThatPrecisionsGet() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        try writeVendorLocal(f)
        let entry = try #require(await f.service.catalog().first { $0.id == "local-voices" }, "vendor folder admitted")
        let model = try await f.service.manifest("local-voices")
        #expect(model.vendor?["int4"]?.repo == "test/vendor")
        #expect(PinnedFilesCheck.baseSources(model).map(\.repo) == ["test/fixture"])
        try await f.service.add(id: "local-voices")
        let lister = Lister(.files(["base.bin"]), byRepo: ["test/vendor": .files([])])
        await f.service.setPinnedFilesLister(lister.lister)
        // Approval checks the base only, so the bad vendor pin does not block it.
        try await f.service.approveLocalCode(id: "local-voices", contentDigest: try #require(entry.row.contentDigest), approvedBy: .mcp)
        do {
            try await f.service.get(id: "local-voices", precision: "int4")
            Issue.record("a Get of a vendor tier the repository lacks went ahead")
        } catch let error as WorkerFailure {
            #expect(error.code == "pinned_file_missing" && error.message.hasPrefix("test/vendor has no file v.bin at revision aaaaaaaaaaaa"), "\(error.message)")
        }
        let row = try #require(await f.service.catalog().first { $0.id == "local-voices" }?.row)
        #expect(row.states["int4"]?.phase == "pinned_file_missing")
        // The folder stays usable: the 16-bit Get passes the pin check (it then downloads from the fixture's server).
        try await f.service.get(id: "local-voices", precision: "bf16")
        #expect(await f.service.catalog().first { $0.id == "local-voices" }?.row.states["bf16"]?.state == .ready)
        await f.service.shutdown()
    }

    // MARK: Hugging Face client (loopback stand-in)

    @Test func listingReadsTheRevisionFileList() async throws {
        let root = URL(fileURLWithPath: "/tmp/arsenal-fetch/listing/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let revision = String(repeating: "a", count: 40)
        let api = root.appendingPathComponent("models/test/listed/revision/\(revision)")
        try FileManager.default.createDirectory(at: api.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(#"{"sha":"\#(revision)","siblings":[{"rfilename":"config.json"},{"rfilename":"model.safetensors"}]}"#.utf8).write(to: api)
        let server = try await StandIn(root: root); defer { server.stop() }
        var client = HuggingFaceClient(baseURL: server.url); client.token = { nil }
        #expect(await client.listing(repo: "test/listed", revision: revision) == .files(["config.json", "model.safetensors"]))
        #expect(await client.listing(repo: "test/absent", revision: revision) == .repoMissing(hasToken: false))
        #expect(await client.listing(repo: "test/gated-meta", revision: revision) == .gated(repo: "test/gated-meta"))
        // An expired token (a bare 401): the listing retries without it and says so.
        var expired = HuggingFaceClient(baseURL: server.url); expired.token = { "hf_TESTONLYexpired0123456789" }; expired.sendsTokenToLoopback = true
        #expect(await expired.listing(repo: "test/bad-token", revision: revision) == .repoMissing(hasToken: false, tokenRejected: true))
        var offline = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:1/")!); offline.token = { nil }
        if case .unreachable = await offline.listing(repo: "test/listed", revision: revision) {} else { Issue.record("a refused connection was not unreachable") }
    }

    @Test func downloadNotFoundIsTerminalNotResumable() async throws {
        let root = URL(fileURLWithPath: "/tmp/arsenal-fetch/missing-pin/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        var client = HuggingFaceClient(baseURL: server.url); client.token = { nil }; client.maximumAttempts = 4
        let revision = String(repeating: "a", count: 40)
        let file = DerivedFile(path: "weights/model.safetensors", bytes: 3, sha256: String(repeating: "b", count: 64))
        let error = await #expect(throws: HFMissingFileError.self) {
            _ = try await client.download(repo: "test/missing", revision: revision, file: file, directory: root.appendingPathComponent("downloads"))
        }
        #expect(error == HFMissingFileError(repo: "test/missing", revision: revision, path: "weights/model.safetensors"))
        #expect(error?.localizedDescription.hasPrefix("weights/model.safetensors is not in the repo test/missing at revision aaaaaaaaaaaa") == true)
        #expect(error?.localizedDescription.contains("resume") == false)
    }
}
