// SPDX-License-Identifier: MIT
import Foundation
import Security
import Testing
@testable import ArsenalCore

/// Gated Hugging Face repositories in the fetch layer: a gate is its own typed error (with or without a token, and the
/// repository's gate mode), every other refusal stays an ordinary failure. Loopback stand-in only; huggingface.co is never contacted.
@Suite struct GatedFetchTests {
    static let payload = Data(String(repeating: "arsenal-test-weights\n", count: 8192).utf8)
    static let revision = String(repeating: "a", count: 40)
    static let secret = "hf_TESTONLYsecretValue0123456789"
    var file: DerivedFile { DerivedFile(path: "weights.bin", bytes: Int64(Self.payload.count), sha256: FileDigest.sha256(Self.payload)) }

    func root() throws -> URL {
        let url = URL(fileURLWithPath: "/tmp/arsenal-fetch/gated/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true); return url
    }
    func client(_ server: StandIn, token: String?) -> HuggingFaceClient {
        var c = HuggingFaceClient(baseURL: server.url); c.deadline = 10; c.maximumAttempts = 1; c.token = { token }; c.sendsTokenToLoopback = true
        return c
    }
    /// The public metadata the client reads for the gate mode.
    func publishGate(_ root: URL, repo: String, gated: String) throws {
        let url = root.appendingPathComponent("models/test/\(repo)")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(#"{"id":"test/\#(repo)","gated":"\#(gated)"}"#.utf8).write(to: url)
    }
    func download(_ c: HuggingFaceClient, repo: String, root: URL) async throws -> URL {
        try await c.download(repo: "test/" + repo, revision: Self.revision, file: file, directory: root.appendingPathComponent("downloads-\(repo)"))
    }

    @Test func noTokenRefusalIsAGateWithoutToken() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        try publishGate(root, repo: "gated", gated: "auto")
        let error = await #expect(throws: HFGateError.self) { try await download(client(server, token: nil), repo: "gated", root: root) }
        #expect(error == HFGateError(repo: "test/gated", hasToken: false, mode: "auto"))
        #expect(error?.pages.map(\.absoluteString) == ["https://huggingface.co/test/gated", "https://huggingface.co/settings/tokens"])
        #expect(try String(contentsOf: root.appendingPathComponent("auth-gated"), encoding: .utf8) == "none")
    }

    @Test func tokenNotAcceptedIsAGateWithToken() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        try publishGate(root, repo: "gated", gated: "manual")
        let error = await #expect(throws: HFGateError.self) { try await download(client(server, token: Self.secret), repo: "gated", root: root) }
        #expect(error == HFGateError(repo: "test/gated", hasToken: true, mode: "manual"))
        #expect(error?.pages.map(\.absoluteString) == ["https://huggingface.co/test/gated"])
        #expect(try String(contentsOf: root.appendingPathComponent("auth-gated"), encoding: .utf8) == "token")
        // The message names the repository and the fix, never the token.
        #expect(error?.localizedDescription.contains(Self.secret) == false)
        #expect(String(describing: error as Any).contains(Self.secret) == false)
    }

    @Test func acceptedRepositoryDownloadsAfterRefusals() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        let c = client(server, token: Self.secret)
        for _ in 0..<2 { await #expect(throws: HFGateError.self) { try await download(c, repo: "gated-after-2", root: root) } }
        #expect(try Data(contentsOf: try await download(c, repo: "gated-after-2", root: root)) == Self.payload)
    }

    @Test(arguments: ["forbidden", "missing", "not-found"]) func otherRefusalsAreNotGates(mode: String) async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        for token in [nil, Self.secret] {
            do {
                _ = try await download(client(server, token: token), repo: mode, root: root); Issue.record("\(mode) accepted")
            } catch {
                #expect(!(error is HFGateError), "\(mode) with token \(token != nil) became a gate")
                #expect(!error.localizedDescription.contains(Self.secret))
            }
        }
    }

    @Test func bareUnauthorizedIsAGateOnlyWithoutToken() async throws {
        #expect(HFGateError.isGate(status: 401, errorCode: nil, sentToken: false))
        #expect(!HFGateError.isGate(status: 401, errorCode: nil, sentToken: true))
        #expect(HFGateError.isGate(status: 403, errorCode: "GatedRepo", sentToken: true))
        #expect(!HFGateError.isGate(status: 403, errorCode: nil, sentToken: true))
        #expect(!HFGateError.isGate(status: 401, errorCode: "RepoNotFound", sentToken: false))
        #expect(!HFGateError.isGate(status: 404, errorCode: "GatedRepo", sentToken: false))
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        await #expect(throws: HFGateError.self) { try await download(client(server, token: nil), repo: "anon401", root: root) }
    }

    @Test func rejectedTokenIsDroppedOnceForAPublicFile() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        #expect(try Data(contentsOf: try await download(client(server, token: Self.secret), repo: "bad-token", root: root)) == Self.payload)
    }

    @Test func aRejectedTokenOnAGatedRepoSaysSo() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        let error = await #expect(throws: HFGateError.self) { try await download(client(server, token: Self.secret), repo: "gated-bad-token", root: root) }
        #expect(error?.hasToken == false && error?.tokenRejected == true)
        #expect(error?.localizedDescription.contains("refused Arsenal's token") == true)
        #expect(error?.pages.last == HFGateError.tokenPage)
    }

    @Test func publicMetadataLoadsDespiteARejectedToken() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        try publishGate(root, repo: "bad-token", gated: "auto")
        #expect(await client(server, token: Self.secret).gateMode(repo: "test/bad-token", repoType: "model") == "auto")
    }

    @Test func aRedirectToAnotherHostCarriesNoToken() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        #expect(await client(server, token: Self.secret).gateMode(repo: "test/redirect-out", repoType: "model") == "auto")
        #expect(try String(contentsOf: root.appendingPathComponent("auth-redirect"), encoding: .utf8) == "none")
        let original = URL(string: "https://huggingface.co/api/models/a/b")!
        var cdn = URLRequest(url: URL(string: "https://cdn-lfs.hf.co/x")!); cdn.setValue("Bearer t", forHTTPHeaderField: "Authorization")
        #expect(HFRedirectGuard.cleaned(cdn, from: original).value(forHTTPHeaderField: "Authorization") == nil)
        var same = URLRequest(url: URL(string: "https://huggingface.co/api/models/a/c")!); same.setValue("Bearer t", forHTTPHeaderField: "Authorization")
        #expect(HFRedirectGuard.cleaned(same, from: original).value(forHTTPHeaderField: "Authorization") == "Bearer t")
    }

    @Test func pinsAndMetadataReportTheGate() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let server = try await StandIn(root: root); defer { server.stop() }
        let weights = WeightSpec(repo: "test/gated-meta", revision: Self.revision, dtype: "fp16", files: ["weights.bin"])
        let pinError = await #expect(throws: HFGateError.self) { _ = try await client(server, token: nil).pins(weights) }
        #expect(pinError?.hasToken == false)
        let metaError = await #expect(throws: HFGateError.self) { _ = try await client(server, token: Self.secret).metadata(repo: "test/gated-meta") }
        #expect(metaError?.hasToken == true)
    }

    @Test func datasetPagesAndTokenOnlyToHuggingFace() {
        #expect(HFGateError(repo: "org/set", repoType: "dataset", hasToken: true).page.absoluteString == "https://huggingface.co/datasets/org/set")
        var loopback = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:9")!); loopback.token = { Self.secret }
        #expect(loopback.requestToken() == nil)
        var hub = HuggingFaceClient(); hub.token = { Self.secret }
        #expect(hub.requestToken() == Self.secret)
    }

    @Test func tokenLookupOrderIsKeychainThenEnvironmentThenCLIFile() throws {
        let home = try root(); defer { try? FileManager.default.removeItem(at: home) }
        let cli = home.appendingPathComponent(".cache/huggingface/token")
        try FileManager.default.createDirectory(at: cli.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("hf_cli\n".utf8).write(to: cli)
        let custom = home.appendingPathComponent("hf-home"); try FileManager.default.createDirectory(at: custom, withIntermediateDirectories: true)
        try Data(" hf_custom ".utf8).write(to: custom.appendingPathComponent("token"))
        #expect(HFTokenSource.resolve(keychain: { "hf_keychain" }, environment: ["HF_TOKEN": "hf_env"], home: home) == "hf_keychain")
        #expect(HFTokenSource.resolve(keychain: { nil }, environment: ["HF_TOKEN": "hf_env"], home: home) == "hf_env")
        #expect(HFTokenSource.resolve(keychain: { nil }, environment: ["HF_TOKEN": "  "], home: home) == "hf_cli")
        #expect(HFTokenSource.resolve(keychain: { nil }, environment: ["HF_HOME": custom.path], home: home) == "hf_custom")
        #expect(HFTokenSource.resolve(keychain: { "" }, environment: [:], home: home.appendingPathComponent("nobody")) == nil)
    }

    /// Token Keychain calls never show the password dialog: an item the `security` command saved (Arsenal not on its
    /// partition list) fails quietly and the lookup falls back, instead of prompting on the user's screen (1 Nov).
    @Test func keychainCallsRunWithoutTheInteractionDialog() {
        var before: DarwinBoolean = false, inside: DarwinBoolean = true, after: DarwinBoolean = false
        SecKeychainGetUserInteractionAllowed(&before)
        let status = HFTokenStore.silently {
            SecKeychainGetUserInteractionAllowed(&inside)
            return errSecAuthFailed
        }
        SecKeychainGetUserInteractionAllowed(&after)
        #expect(status == errSecAuthFailed)
        #expect(!inside.boolValue, "no dialog during the call")
        #expect(after.boolValue == before.boolValue, "restored afterwards")
    }
}
