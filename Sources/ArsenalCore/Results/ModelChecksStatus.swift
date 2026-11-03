// SPDX-License-Identifier: MIT
import Foundation

public enum CheckStatus: String, Codable, Sendable { case pass, fail; case notRun = "not-run" }
/// Evidence is retained separately from timing. A stale pass is never a current pass.
public struct ModelChecksStatus: Codable, Sendable, Equatable {
    public var protocolStatus: CheckStatus
    public var parity: CheckStatus
    public var task: CheckStatus
    public var speed: CheckStatus
    public var stale: Bool
    public var reference: JSONObject?
    public var implementationFingerprint: String
    public var folderDigest: String
    public var fixtureIdentity: String
    public var cases: [String: Int]
    public var evidence: [String]
    public var evidenceHashes: [String: String]
    enum CodingKeys: String, CodingKey {
        case protocolStatus = "protocol", parity, task, speed, stale, reference, cases, evidence
        case evidenceHashes = "evidence_hashes"
        case implementationFingerprint = "implementation_fingerprint", folderDigest = "folder_digest", fixtureIdentity = "fixture_identity"
    }
    public init(
        protocolStatus: CheckStatus = .notRun, parity: CheckStatus = .notRun, task: CheckStatus = .notRun,
        stale: Bool = false, reference: JSONObject? = nil, implementationFingerprint: String = "", folderDigest: String = "",
        fixtureIdentity: String = "", cases: [String: Int] = [:], evidence: [String] = []
    ) {
        self.protocolStatus = protocolStatus; self.parity = parity; self.task = task; self.stale = stale; self.reference = reference
        self.implementationFingerprint = implementationFingerprint; self.folderDigest = folderDigest; self.fixtureIdentity = fixtureIdentity
        self.cases = cases; self.evidence = evidence; self.speed = .notRun; self.evidenceHashes = [:]
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        protocolStatus = try c.decodeIfPresent(CheckStatus.self, forKey: .protocolStatus) ?? .notRun
        parity = try c.decodeIfPresent(CheckStatus.self, forKey: .parity) ?? .notRun
        task = try c.decodeIfPresent(CheckStatus.self, forKey: .task) ?? .notRun
        speed = try c.decodeIfPresent(CheckStatus.self, forKey: .speed) ?? .notRun
        stale = try c.decodeIfPresent(Bool.self, forKey: .stale) ?? false
        reference = try c.decodeIfPresent(JSONObject.self, forKey: .reference)
        implementationFingerprint = try c.decodeIfPresent(String.self, forKey: .implementationFingerprint) ?? ""
        folderDigest = try c.decodeIfPresent(String.self, forKey: .folderDigest) ?? ""
        fixtureIdentity = try c.decodeIfPresent(String.self, forKey: .fixtureIdentity) ?? ""
        cases = try c.decodeIfPresent([String: Int].self, forKey: .cases) ?? [:]
        evidence = try c.decodeIfPresent([String].self, forKey: .evidence) ?? []
        evidenceHashes = try c.decodeIfPresent([String: String].self, forKey: .evidenceHashes) ?? [:]
    }
    public var label: String {
        if stale { return "Stale" }
        if [protocolStatus, parity, task, speed].contains(.fail) { return "Failed" }
        return protocolStatus == .pass ? "Quick check passed" : "Not checked"
    }
}
