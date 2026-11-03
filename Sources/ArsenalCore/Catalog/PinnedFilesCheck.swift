// SPDX-License-Identifier: MIT
import Foundation

/// What Hugging Face says about one repository at one pinned revision.
public enum PinnedFilesListing: Sendable, Equatable {
    /// Every file path at that revision.
    case files(Set<String>)
    /// No such repository or revision (or a private one the token cannot see). `tokenRejected`: Hugging Face refused the
    /// token Arsenal found (expired or revoked), so the listing went without one.
    case repoMissing(hasToken: Bool, tokenRejected: Bool = false)
    /// The repository is gated and refused the listing: the Get's gated wait handles it, not this check.
    case gated(repo: String)
    /// Hugging Face could not be reached or answered with something else; the reason is shown, never a token.
    case unreachable(String)
}
/// Lists a repository's files at a revision. The app passes Hugging Face; tests pass a stand-in (no network).
public typealias PinnedFilesLister = @Sendable (_ repo: String, _ revision: String, _ repoType: String) async -> PinnedFilesListing

/// Whether every weights file a local folder pins exists in its Hugging Face repository at the pinned revision. Run once
/// per folder content digest, before approval and Get, so a pin Hugging Face lacks becomes an error row instead of a
/// download that fails with a 404 on every retry.
public enum PinnedFilesCheck: Sendable, Equatable {
    case present
    case missing(repo: String, revision: String, files: [String])
    case repoMissing(repo: String, revision: String, hasToken: Bool, tokenRejected: Bool = false)
    /// The listing was refused by the gate: the pins are unchecked, and the Get waits on the gate as usual.
    case gated(repo: String)
    /// Hugging Face could not be reached: the pins are unchecked and approval goes ahead (the download says so if a pin is wrong).
    case unreachable(repo: String, reason: String)

    /// The folder must not be approved or downloaded: the row shows this.
    public var problem: String? {
        switch self {
        case .missing(let repo, let revision, let files):
            let one = files.count == 1
            return "\(repo) has no \(one ? "file" : "files") \(files.joined(separator: ", ")) at revision \(revision.prefix(12)): model.toml pins "
                + "\(one ? "a file" : "files") Hugging Face does not have, so Get can never download \(one ? "it" : "them"). "
                + "Fix the pin in model.toml (kit check lists the repo's files)."
        case .repoMissing(let repo, let revision, let hasToken, let tokenRejected):
            var privately = ""
            if !hasToken {
                privately =
                    tokenRejected
                    ? ", or it is private and Hugging Face refused Arsenal's token (expired or revoked; add a new one in Settings)"
                    : ", or it is private and Arsenal has no Hugging Face token"
            }
            return "Hugging Face has no repository \(repo) at revision \(revision.prefix(12))\(privately): fix weights.repo and weights.revision in model.toml."
        default: return nil
        }
    }
    /// Approval goes ahead without the check: said once, with the approval.
    public var note: String? {
        switch self {
        case .unreachable(let repo, let reason):
            return "Arsenal could not reach Hugging Face to check that \(repo) has the pinned files (\(reason)), so approval went ahead without that check; "
                + "a wrong pin would show at download."
        case .gated(let repo):
            return "\(repo) is gated on Hugging Face, so its files could not be listed; Get will wait until its terms are accepted."
        default: return nil
        }
    }
    /// Unreachable is retried on the next approval or Get, and so is a missing repository seen without a working token (a
    /// token saved later may show it); every other result holds for the folder's digest.
    public var isFinal: Bool {
        switch self {
        case .unreachable: false
        case .repoMissing(_, _, let hasToken, _): hasToken
        default: true
        }
    }

    /// What discovery and approval check: the base weights and the assets the base download needs. A vendor precision's
    /// release is checked only by the Get of that precision (`run(sources:)` with what that Get downloads).
    public static func baseSources(_ model: ModelManifest) -> [WeightSpec] { [model.weights] + model.selectedAssets.map(\.weights) }
    public static func run(_ model: ModelManifest, lister: PinnedFilesLister) async -> PinnedFilesCheck {
        await run(sources: baseSources(model), lister: lister)
    }
    public static func run(sources: [WeightSpec], lister: PinnedFilesLister) async -> PinnedFilesCheck {
        var grouped: [(key: String, repo: String, revision: String, type: String, files: [String])] = []
        let checkable = sources.filter { !$0.files.isEmpty && ModelManifest.isHex($0.revision, count: 40) && $0.repo.split(separator: "/").count == 2 }
        for source in checkable {
            let type = source.repoType ?? "model", key = "\(type):\(source.repo)@\(source.revision)"
            if let index = grouped.firstIndex(where: { $0.key == key }) {
                grouped[index].files += source.files.filter { !grouped[index].files.contains($0) }
            } else {
                grouped.append((key, source.repo, source.revision, type, source.files))
            }
        }
        var softer: PinnedFilesCheck?
        for group in grouped {
            switch await lister(group.repo, group.revision, group.type) {
            case .files(let present):
                let missing = group.files.filter { !present.contains($0) }
                if !missing.isEmpty { return .missing(repo: group.repo, revision: group.revision, files: missing) }
            case .repoMissing(let hasToken, let rejected):
                return .repoMissing(repo: group.repo, revision: group.revision, hasToken: hasToken, tokenRejected: rejected)
            case .gated(let repo): softer = softer ?? .gated(repo: repo)
            case .unreachable(let reason): softer = .unreachable(repo: group.repo, reason: reason)
            }
        }
        return softer ?? .present
    }
}

/// A pinned file Hugging Face answers 404 for at its pinned revision: no retry can download it.
public struct HFMissingFileError: Error, LocalizedError, Sendable, Equatable {
    public var repo: String; public var revision: String; public var path: String
    public init(repo: String, revision: String, path: String) { self.repo = repo; self.revision = revision; self.path = path }
    public var errorDescription: String? {
        "\(path) is not in the repo \(repo) at revision \(revision.prefix(12)) (Hugging Face answers 404), so another Get cannot download it. The model's pins must change."
    }
}

extension HuggingFaceClient {
    /// The repository's files at `revision`, with the same token handling as Get (Keychain, HF_TOKEN, CLI token file).
    public func listing(repo: String, revision: String, repoType: String = "model") async -> PinnedFilesListing {
        guard ModelManifest.isHex(revision, count: 40), repo.split(separator: "/").count == 2, SafePath.isRelative(repo) else { return .repoMissing(hasToken: false) }
        var url = baseURL.appendingPathComponent(repoType == "dataset" ? "api/datasets" : "api/models")
        for part in repo.split(separator: "/") { url.appendPathComponent(String(part)) }
        url.appendPathComponent("revision"); url.appendPathComponent(revision)
        var request = URLRequest(url: url); request.timeoutInterval = 15
        let reply: (data: Data, reply: URLResponse, sentToken: Bool, tokenRejected: Bool)
        do { reply = try await data(request) } catch { return .unreachable((error as? URLError).map { Self.reason($0) } ?? "no answer") }
        guard let http = reply.reply as? HTTPURLResponse else { return .unreachable("no HTTP answer") }
        let code = http.value(forHTTPHeaderField: "X-Error-Code")
        if HFGateError.isGate(status: http.statusCode, errorCode: code, sentToken: reply.sentToken) { return .gated(repo: repo) }
        if http.statusCode == 404 || ["RepoNotFound", "RevisionNotFound"].contains(code ?? "") {
            return .repoMissing(hasToken: reply.sentToken, tokenRejected: reply.tokenRejected)
        }
        guard http.statusCode == 200 else { return .unreachable("HTTP \(http.statusCode)") }
        guard let object = try? JSONSerialization.jsonObject(with: reply.data) as? [String: Any], let siblings = object["siblings"] as? [[String: Any]] else {
            return .unreachable("an unreadable file list")
        }
        return .files(Set(siblings.compactMap { $0["rfilename"] as? String }))
    }
    static func reason(_ error: URLError) -> String {
        switch error.code {
        case .notConnectedToInternet, .networkConnectionLost, .dataNotAllowed: "offline"
        case .timedOut: "timed out"
        case .cannotFindHost, .dnsLookupFailed, .cannotConnectToHost: "huggingface.co unreachable"
        default: "network error \(error.code.rawValue)"
        }
    }
}
