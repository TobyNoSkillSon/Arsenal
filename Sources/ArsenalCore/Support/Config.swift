// SPDX-License-Identifier: MIT
import Foundation

public struct ModelPreferences: Codable, Sendable {
    public var precision: String; public var fast: Bool; public var keepHot: Bool; public var toolsShown: Bool = true; public var explicitSelection: Bool = true
    public init(precision: String, fast: Bool = true, keepHot: Bool = false, explicitSelection: Bool = true) {
        self.precision = precision; self.fast = fast; self.keepHot = keepHot; self.explicitSelection = explicitSelection
    }
    enum CodingKeys: String, CodingKey { case precision, fast, keepHot, toolsShown, explicitSelection }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        precision = try c.decode(String.self, forKey: .precision)
        fast = try c.decodeIfPresent(Bool.self, forKey: .fast) ?? true
        keepHot = try c.decodeIfPresent(Bool.self, forKey: .keepHot) ?? false
        toolsShown = try c.decodeIfPresent(Bool.self, forKey: .toolsShown) ?? true
        explicitSelection = try c.decodeIfPresent(Bool.self, forKey: .explicitSelection) ?? true
    }
}
/// Last signed display identity, retained independently of disposable installed descriptors.
public struct CatalogModelIdentity: Codable, Sendable, Equatable {
    public var name: String
    public var kind: String
}
public struct ArsenalConfiguration: Codable, Sendable {
    public var catalogModelIdentities: [String: CatalogModelIdentity] = [:]
    public var localCodeApprovals: [String: CodeApproval] = [:]
    public var localCodeConsent: [String: String] = [:]
    public var rejectedRevisions: [String: String] = [:]
    public var migrations: JSONObject = [:]
    public private(set) var warnings: [String] = []
    var legacyTableNeedsMigration = false
    public var table: [String] = []
    public var settings: [String: JSONObject] = [:]
    public var runnerOverrides: [String: String] = [:]
    public var selections: [String: ModelPreferences] = [:]
    public var residency = ResidencySettings()
    public var memory = MemorySettings()
    public var kindOrder: [String] = []
    /// One-shot explicit save intent; decoded configurations never retain it.
    var requestedKindOrder: [String]?
    public var catalog: JSONObject = [:]
    public var get: JSONObject = [:]
    public init() {}
    enum CodingKeys: String, CodingKey {
        case catalogModelIdentities, rejectedRevisions, memory, runnerOverrides, migrations, localCodeApprovals, localCodeConsent, table, settings, selections, residency,
            kindOrder, catalog,
            get
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        catalogModelIdentities = try c.decodeIfPresent([String: CatalogModelIdentity].self, forKey: .catalogModelIdentities) ?? [:]
        runnerOverrides = try c.decodeIfPresent([String: String].self, forKey: .runnerOverrides) ?? [:]
        localCodeApprovals = try c.decodeIfPresent([String: CodeApproval].self, forKey: .localCodeApprovals) ?? [:]
        localCodeConsent = try c.decodeIfPresent([String: String].self, forKey: .localCodeConsent) ?? [:]
        rejectedRevisions = try c.decodeIfPresent([String: String].self, forKey: .rejectedRevisions) ?? [:]
        migrations = try c.decodeIfPresent(JSONObject.self, forKey: .migrations) ?? [:]
        legacyTableNeedsMigration = !c.contains(.table) && migrations["tableV2"]?.bool != true
        table = try c.decodeIfPresent([String].self, forKey: .table) ?? []
        settings = try c.decodeIfPresent([String: JSONObject].self, forKey: .settings) ?? [:]
        selections = try c.decodeIfPresent([String: ModelPreferences].self, forKey: .selections) ?? [:]
        memory = try c.decodeIfPresent(MemorySettings.self, forKey: .memory) ?? MemorySettings()
        residency = try c.decodeIfPresent(ResidencySettings.self, forKey: .residency) ?? ResidencySettings()
        kindOrder = try c.decodeIfPresent([String].self, forKey: .kindOrder) ?? []
        catalog = try c.decodeIfPresent(JSONObject.self, forKey: .catalog) ?? [:]
        get = try c.decodeIfPresent(JSONObject.self, forKey: .get) ?? [:]
    }
    public static func read(_ path: URL) throws -> Self {
        guard FileManager.default.fileExists(atPath: path.path) else { return Self() }
        do { return try JSONDecoder().decode(Self.self, from: Data(contentsOf: path)) } catch { throw unreadableConfiguration() }
    }
    /// Apply only changes since the service's last read to the newest disk object. Unknown keys survive.
    /// Validate before the atomic write so a refused save cannot partially commit state.
    func writeMerging(_ path: URL, baseline: Self) throws -> Self {
        let previous = try WireJSON.decode(JSONEncoder().encode(baseline))
        let candidate = try WireJSON.decode(JSONEncoder().encode(self))
        let disk: JSONObject
        do { disk = FileManager.default.fileExists(atPath: path.path) ? try WireJSON.decode(Data(contentsOf: path)) : [:] } catch { throw Self.unreadableConfiguration() }
        var merged = Self.mergeChanges(previous: previous, candidate: candidate, disk: disk)
        for key in ["table"] where previous[key] != candidate[key] {
            let old = previous[key]?.array?.compactMap(\.string) ?? []
            let new = candidate[key]?.array?.compactMap(\.string) ?? []
            let outside = disk[key]?.array?.compactMap(\.string) ?? []
            merged[key] = .array(Self.mergeList(previous: old, candidate: new, disk: outside).map(JSONValue.string))
        }
        // An explicit group-order save replaces the list even if it equals our stale baseline.
        // Membership merging is for the model table, never for the user's requested positions.
        if let requestedKindOrder { merged["kindOrder"] = .array(requestedKindOrder.map(JSONValue.string)) }
        // An external deletion followed by a new edit of that model recreates only that model's required dtype.
        var selectionsObject = merged["selections"]?.object ?? [:]
        for id in selectionsObject.keys where selectionsObject[id]?["precision"] == nil {
            if let dtype = selections[id]?.precision, var preference = selectionsObject[id]?.object {
                preference["precision"] = .string(dtype); selectionsObject[id] = .object(preference)
            }
        }
        for id in selectionsObject.keys {
            if var preference = selectionsObject[id]?.object { preference["fast"] = true; selectionsObject[id] = .object(preference) }
        }
        if !selectionsObject.isEmpty { merged["selections"] = .object(selectionsObject) }
        // Retire the old app appearance preference silently; macOS owns appearance now.
        merged["appearance"] = nil
        let data = try WireJSON.encode(merged)
        let validated: Self
        do { validated = try JSONDecoder().decode(Self.self, from: data) } catch { throw Self.unreadableConfiguration() }
        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: path, options: .atomic)
        return validated
    }
    private static func unreadableConfiguration() -> WorkerFailure {
        WorkerFailure(code: "config", message: "Arsenal could not read config.json. Repair the file and try again.")
    }
    static func mergeList(previous: [String], candidate: [String], disk: [String]) -> [String] {
        let removed = Set(previous).subtracting(candidate), added = candidate.filter { !previous.contains($0) }
        var seen = Set<String>()
        var merged = (disk.filter { !removed.contains($0) } + added).filter { seen.insert($0).inserted }
        // Retain outside additions in their slots; apply an explicit app reorder only to shared members.
        if previous.filter(candidate.contains) != candidate.filter(previous.contains) {
            var reordered = candidate.filter(merged.contains).makeIterator()
            for index in merged.indices where candidate.contains(merged[index]) { merged[index] = reordered.next()! }
        }
        return merged
    }
    static func mergeChanges(previous: JSONObject, candidate: JSONObject, disk: JSONObject) -> JSONObject {
        var merged = disk
        for key in Set(previous.keys).union(candidate.keys) where previous[key] != candidate[key] {
            if let new = candidate[key]?.object {
                merged[key] = .object(mergeChanges(previous: previous[key]?.object ?? [:], candidate: new, disk: disk[key]?.object ?? [:]))
            } else {
                merged[key] = candidate[key]
            }
        }
        return merged
    }
    /// Decoding warnings describe values that a successful write has normalized.
    mutating func clearWarningsAfterWrite() { warnings.removeAll() }
    /// Preserve app/API-owned keys rather than replacing the whole shared configuration.
    public func write(_ path: URL) throws {
        var object: JSONObject = [:]
        if FileManager.default.fileExists(atPath: path.path) { object = try WireJSON.decode(Data(contentsOf: path)) }
        object["appearance"] = nil
        let own = try WireJSON.decode(JSONEncoder().encode(self))
        for key in [
            "catalogModelIdentities", "rejectedRevisions", "runnerOverrides", "migrations", "localCodeApprovals", "localCodeConsent", "table", "settings", "selections",
            "residency", "memory",
            "kindOrder"
        ] {
            object[key] = own[key]
        }
        var savedCatalog = object["catalog"]?.object ?? [:]
        if savedCatalog["source"] == nil, let source = catalog["source"] { savedCatalog["source"] = source }
        var savedSource = savedCatalog["source"]?.object ?? [:]
        for (key, value) in catalog["source"]?.object ?? [:] where savedSource[key] == nil { savedSource[key] = value }
        if !savedSource.isEmpty { savedCatalog["source"] = .object(savedSource) }
        object["catalog"] = .object(savedCatalog)
        if object["get"] == nil && !get.isEmpty { object["get"] = .object(get) }
        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        try WireJSON.encode(object).write(to: path, options: .atomic)
    }
}
