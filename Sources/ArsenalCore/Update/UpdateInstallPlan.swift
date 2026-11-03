// SPDX-License-Identifier: MIT
import Foundation

public struct UpdateInstallPlan: Codable, Sendable {
    public let staged: StagedArsenalUpdate
    public let destination: URL
    public let previousVersion: String
    public let parentPID: Int32
    public let support: URL
    public let loadedModels: [String]
    public let token: String
    public init(staged: StagedArsenalUpdate, destination: URL, previousVersion: String, parentPID: Int32, support: URL, loadedModels: [String]) {
        self.staged = staged; self.destination = destination; self.previousVersion = previousVersion; self.parentPID = parentPID
        self.support = support; self.loadedModels = loadedModels; token = UUID().uuidString
    }
    public func persistRestore(version: String) throws {
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try JSONEncoder().encode(UpdateRestore(token: token, version: version, models: loadedModels)).write(
            to: support.appendingPathComponent(UpdateRestore.fileName), options: .atomic)
    }
}
struct UpdateRestore: Codable {
    static let fileName = "update-restore.json"
    let token: String
    let version: String
    let models: [String]
    static func remove(support: URL, token: String) {
        let url = support.appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: url), let record = try? JSONDecoder().decode(Self.self, from: data), record.token == token else { return }
        try? FileManager.default.removeItem(at: url)
    }
}
public struct UpdateAcknowledgement: Codable, Sendable {
    public let token: String
    public let pid: Int32
    public enum Phase: String, Codable, Sendable { case launched, restored }
    public let phase: Phase
    public let reason: String?
    public init(token: String, pid: Int32, phase: Phase, reason: String? = nil) {
        self.token = token; self.pid = pid; self.phase = phase; self.reason = reason
    }
    public static func url(support: URL, token: String) -> URL { support.appendingPathComponent("update-ack-\(token).json") }
}
