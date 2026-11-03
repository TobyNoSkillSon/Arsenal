// SPDX-License-Identifier: MIT
import Foundation
import Security

/// A Hugging Face repository that refuses its files until the signed-in account accepts the owner's terms ("gated").
/// Distinct from every other download failure: the fix is a person's click on the model's page, not a retry of the bytes.
/// Its text never carries a token.
public struct HFGateError: Error, LocalizedError, Sendable, Equatable {
    public var repo: String
    public var repoType: String
    /// A token was sent with the refused request. False: Arsenal has no Hugging Face token at all.
    public var hasToken: Bool
    /// The repository's `gated` field: "auto" (access on the click), "manual" (the owner approves each request), nil unknown.
    public var mode: String?
    /// Hugging Face refused the token Arsenal found (expired or revoked), so the request went without one.
    public var tokenRejected: Bool
    public init(repo: String, repoType: String = "model", hasToken: Bool, mode: String? = nil, tokenRejected: Bool = false) {
        self.repo = repo; self.repoType = repoType; self.hasToken = hasToken; self.mode = mode; self.tokenRejected = tokenRejected
    }
    /// The page where a person accepts the terms (always the public site, whatever base URL the client used).
    public var page: URL {
        var url = URL(string: "https://huggingface.co")!
        if repoType == "dataset" { url.appendPathComponent("datasets") }
        for part in repo.split(separator: "/") { url.appendPathComponent(String(part)) }
        return url
    }
    public static let tokenPage = URL(string: "https://huggingface.co/settings/tokens")!
    /// The pages a person needs. Without a token the token page opens last, so it is the tab in front (it comes first).
    public var pages: [URL] { hasToken ? [page] : [page, Self.tokenPage] }
    public var errorDescription: String? {
        if hasToken { return "\(repo) is gated on Hugging Face: accept its terms on the model's page, then Get again." }
        if tokenRejected {
            return "\(repo) is gated on Hugging Face and it refused Arsenal's token (expired or revoked): add a new one in Settings, then accept the model's terms on its page."
        }
        return "\(repo) is gated on Hugging Face and Arsenal has no Hugging Face token: add one in Settings, then accept the model's terms on its page."
    }

    /// Whether a refused response is a gate. `GatedRepo` is Hugging Face's own code; a bare 401 to a request sent without
    /// a token is one too (the Hub answers an anonymous request for a gated file that way). Any other 401/403 (a missing or
    /// private repository, `RepoNotFound`, a bad token) and every 404 is an ordinary failure.
    public static func isGate(status: Int, errorCode: String?, sentToken: Bool) -> Bool {
        if errorCode == "GatedRepo" { return [401, 403].contains(status) }
        return status == 401 && !sentToken && (errorCode ?? "").isEmpty
    }
}

/// Where Arsenal finds a Hugging Face read token, in order: its own Keychain item, then `HF_TOKEN`, then the Hugging Face
/// CLI's token file (`$HF_HOME/token`, else `~/.cache/huggingface/token`). Tokens are never logged or printed.
public enum HFTokenSource {
    public static func resolve(
        keychain: () -> String? = { HFTokenStore.read() }, environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> String? {
        if let token = clean(keychain()) { return token }
        if let token = clean(environment["HF_TOKEN"]) { return token }
        let file =
            environment["HF_HOME"].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0).appendingPathComponent("token") }
            ?? home.appendingPathComponent(".cache/huggingface/token")
        return clean((try? Data(contentsOf: file, options: .uncached)).flatMap { String(data: $0.prefix(4096), encoding: .utf8) })
    }
    /// One line, no whitespace; nil when empty.
    static func clean(_ value: String?) -> String? {
        guard let line = value?.trimmingCharacters(in: .whitespacesAndNewlines), !line.isEmpty, !line.contains(where: \.isWhitespace) else { return nil }
        return line
    }
}

extension HFTokenStore {
    static let service = "dev.arsenal.huggingface", account = "read-token"
    /// Saves the read token in the login Keychain (replacing any earlier one). Returns the Keychain status; never logs the value.
    @discardableResult public static func write(_ token: String) -> OSStatus { write(token, service: service) }
    /// Forgets the saved token. Success when there was none.
    @discardableResult public static func delete() -> OSStatus { delete(service: service) }
    public static var hasStoredToken: Bool { read() != nil }
    static func write(_ token: String, service: String) -> OSStatus {
        guard let value = HFTokenSource.clean(token) else { return errSecParam }
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account]
        let data = Data(value.utf8)
        let status = SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        guard status == errSecItemNotFound else { return status }
        var item = query
        item[kSecValueData as String] = data; item[kSecAttrLabel as String] = "Arsenal Hugging Face token"
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(item as CFDictionary, nil)
    }
    static func delete(service: String) -> OSStatus {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecItemNotFound ? errSecSuccess : status
    }
}

extension HuggingFaceClient {
    /// The token for a request to this client's host, or nil: a token goes only to huggingface.co (and, in tests that ask
    /// for it, to the loopback stand-in).
    func requestToken() -> String? {
        let host = baseURL.host ?? ""
        guard host == "huggingface.co" || (sendsTokenToLoopback && ["127.0.0.1", "localhost"].contains(host)) else { return nil }
        return token()
    }
    /// The gate error for a refused request: reads the repository's public `gated` field once (best effort, nil on any failure).
    func gateError(repo: String, repoType: String, sentToken: Bool, tokenRejected: Bool = false) async -> HFGateError {
        HFGateError(
            repo: repo, repoType: repoType, hasToken: sentToken, mode: await gateMode(repo: repo, repoType: repoType), tokenRejected: !sentToken && tokenRejected)
    }
    /// A small request with the token (when there is one); a bare 401 to the token (expired, revoked) is retried once
    /// without it, so public metadata still loads. `sentToken`: the reply answered a request that carried one.
    func data(_ request: URLRequest) async throws -> (data: Data, reply: URLResponse, sentToken: Bool, tokenRejected: Bool) {
        var request = request
        guard let token = requestToken() else { let (d, r) = try await HFRedirectGuard.session.data(for: request); return (d, r, false, false) }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let (data, reply) = try await HFRedirectGuard.session.data(for: request)
        guard let http = reply as? HTTPURLResponse, http.statusCode == 401, (http.value(forHTTPHeaderField: "X-Error-Code") ?? "").isEmpty else {
            return (data, reply, true, false)
        }
        request.setValue(nil, forHTTPHeaderField: "Authorization")
        let (again, second) = try await HFRedirectGuard.session.data(for: request)
        return (again, second, false, true)
    }
    func gateMode(repo: String, repoType: String) async -> String? {
        var url = baseURL.appendingPathComponent(repoType == "dataset" ? "api/datasets" : "api/models")
        for part in repo.split(separator: "/") { url.appendPathComponent(String(part)) }
        var request = URLRequest(url: url); request.timeoutInterval = 10
        guard let (data, reply, _, _) = try? await self.data(request), (reply as? HTTPURLResponse)?.statusCode == 200,
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return object["gated"] as? String
    }
    /// Throws the gate error when a metadata or pin reply is a gate refusal; other replies pass.
    func throwIfGated(_ reply: URLResponse, repo: String, repoType: String, sentToken: Bool, tokenRejected: Bool = false) async throws {
        guard let http = reply as? HTTPURLResponse,
            HFGateError.isGate(status: http.statusCode, errorCode: http.value(forHTTPHeaderField: "X-Error-Code"), sentToken: sentToken)
        else { return }
        throw await gateError(repo: repo, repoType: repoType, sentToken: sentToken, tokenRejected: tokenRejected)
    }
}

/// Small Hub requests (metadata, pins, gate mode): a redirect to another host (a CDN) never carries the token, as in
/// HuggingFaceClient's transfers.
final class HFRedirectGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    static let session = URLSession(configuration: .ephemeral, delegate: HFRedirectGuard(), delegateQueue: nil)
    func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(Self.cleaned(request, from: task.originalRequest?.url))
    }
    static func cleaned(_ request: URLRequest, from original: URL?) -> URLRequest {
        var clean = request
        if request.url?.host != original?.host || request.url?.port != original?.port { clean.setValue(nil, forHTTPHeaderField: "Authorization") }
        return clean
    }
}
