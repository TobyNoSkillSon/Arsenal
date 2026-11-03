// SPDX-License-Identifier: MIT
import Foundation

public struct GetPlanItem: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var kind: String
    /// Nil is an unpinned dependency estimate, not a zero-byte download.
    public var downloadBytes: Int64?
    public var localCopyBytes: Int64
    /// The part of `localCopyBytes` that clones on the store's volume and uses no disk; nil when not known (counts as copied).
    public var localCloneBytes: Int64?
    public var networkContent: [String: Int64]?
    public init(
        id: String, name: String, kind: String, downloadBytes: Int64?, localCopyBytes: Int64 = 0, networkContent: [String: Int64]? = nil, localCloneBytes: Int64? = nil
    ) {
        self.id = id; self.name = name; self.kind = kind; self.downloadBytes = downloadBytes; self.localCopyBytes = localCopyBytes; self.networkContent = networkContent
        self.localCloneBytes = localCloneBytes
    }
}
public struct GetPlan: Codable, Sendable, Equatable {
    public var id: String
    public var dtype: String
    public var runner: String
    public var items: [GetPlanItem]
    public var identity: String?
    public var totalDownloadBytes: Int64? {
        var total: Int64 = 0
        for item in items {
            guard let bytes = item.downloadBytes else { return nil }; let next = total.addingReportingOverflow(bytes); guard !next.overflow else { return nil }
            total = next.partialValue
        }
        return total
    }
    public var requiresDownload: Bool { totalDownloadBytes.map { $0 > 0 } ?? true }
    public init(id: String, dtype: String, runner: String, items: [GetPlanItem], identity: String? = nil) {
        self.id = id; self.dtype = dtype; self.runner = runner; self.items = items; self.identity = identity
    }
    enum CodingKeys: String, CodingKey { case id, dtype, runner, items, identity, totalDownloadBytes, requiresDownload }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try c.decode(String.self, forKey: .id), dtype: try c.decode(String.self, forKey: .dtype), runner: try c.decode(String.self, forKey: .runner),
            items: try c.decode([GetPlanItem].self, forKey: .items), identity: try c.decodeIfPresent(String.self, forKey: .identity))
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id); try c.encode(dtype, forKey: .dtype); try c.encode(runner, forKey: .runner); try c.encode(items, forKey: .items)
        try c.encodeIfPresent(identity, forKey: .identity)
        try c.encode(totalDownloadBytes, forKey: .totalDownloadBytes); try c.encode(requiresDownload, forKey: .requiresDownload)
    }
}
public struct LoadedMenuSection: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var modelID: String
    public var title: String
    public var items: [LoadedAction]
    public init(id: String, modelID: String, title: String, items: [LoadedAction]) { self.id = id; self.modelID = modelID; self.title = title; self.items = items }
}
public struct GetPlanChanged: Error, Sendable, LocalizedError {
    public var revisedPlan: GetPlan
    public var errorDescription: String? { "Get needs a revised download plan; review it before retrying." }
    public init(revisedPlan: GetPlan) { self.revisedPlan = revisedPlan }
}
extension GetPlan {
    public func requireKnownDownloadSize() throws {
        guard let bytes = totalDownloadBytes, bytes >= 0 else {
            let unknown = items.filter { $0.downloadBytes == nil || ($0.downloadBytes ?? 0) < 0 }.map(\.name).joined(separator: ", ")
            throw WorkerFailure(code: "download_size", message: "Download size is not pinned for \(unknown.isEmpty ? id : unknown). Fix the model folder and Refresh before Get.")
        }
    }
    public func covers(_ current: GetPlan) -> Bool {
        guard id == current.id, dtype == current.dtype, runner == current.runner, identity == current.identity else { return false }
        return current.items.allSatisfy { item in
            guard let allowed = items.first(where: { $0.id == item.id && $0.kind == item.kind }),
                let bytes = item.downloadBytes, let limit = allowed.downloadBytes, bytes >= 0, bytes <= limit
            else { return false }
            return (item.networkContent ?? [:]).allSatisfy { allowed.networkContent?[$0.key] == $0.value }
        }
    }
}
extension ModelsService {
    public func get(id: String, precision: String, approvedPlan: GetPlan) async throws {
        let current = try await getPlan(id: id, dtype: precision)
        try current.requireKnownDownloadSize()
        guard approvedPlan.covers(current) else { throw GetPlanChanged(revisedPlan: current) }
        try await get(id: id, precision: precision)
    }
    public func refresh() async throws {}
    public func getPlan(id: String, dtype: String) async throws -> GetPlan {
        guard let row = await catalog(query: .init()).first(where: { $0.id == id })?.row else { throw WorkerFailure(code: "not_found", message: "Unknown model \(id).") }
        guard row.precisions.contains(dtype) else { throw WorkerFailure(code: "precision", message: "That dtype is not offered by \(id).") }
        if let reason = row.dtypeUnavailableReason(dtype) { throw WorkerFailure(code: "unsupported_format", message: reason) }
        return GetPlan(
            id: id, dtype: dtype, runner: row.runner,
            items: [.init(id: "weights", name: row.name + " weights", kind: "weights", downloadBytes: row.downloadBytes[dtype], localCopyBytes: row.localCopyBytes[dtype] ?? 0)])
    }
    public func menuSections() async -> [LoadedMenuSection] {
        let actions = await menuActions()
        return Dictionary(grouping: actions, by: \.modelID).keys.sorted().map { id in .init(id: id, modelID: id, title: id, items: actions.filter { $0.modelID == id }) }
    }
}
