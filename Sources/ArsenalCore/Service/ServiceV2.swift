// SPDX-License-Identifier: MIT
import Foundation

public enum CatalogSort: String, Codable, Sendable, CaseIterable {
    case name, kind, size, ram, popularity, likes, params, publisher, licence, released, added, installed, loaded
}
public struct CatalogQuery: Codable, Sendable, Equatable {
    public var text = ""
    public var kinds: [String] = [], licences: [String] = [], languages: [String] = [], tasks: [String] = [], dtypes: [String] = []
    public var modalitiesIn: [String] = [], modalitiesOut: [String] = []
    public var commercial: Bool?, suggested: Bool?, added: Bool?, installed: Bool?, loaded: Bool?, fitsRAM: Bool?
    public var minSizeBytes: Int64?, maxSizeBytes: Int64?
    public var minRAMMB: Double?, maxRAMMB: Double?
    public var sort: CatalogSort = .name
    public var descending = false
    public init(text: String = "", kinds: [String] = [], sort: CatalogSort = .name, descending: Bool = false) {
        self.text = text; self.kinds = kinds; self.sort = sort; self.descending = descending
    }
}
public struct CatalogEntry: Codable, Sendable, Equatable, Identifiable {
    public var description: String
    public var links: [String: String]
    public var runners: [CatalogRunner]
    public var settingsSummary: [SettingState]
    public var released: Int? { row.released }
    public var row: ModelRow
    public var id: String { row.id }; public var name: String { row.name }; public var kind: String { row.kind }
    public var local: Bool { row.local }
    public var tools: [String] { row.tools }
    public var toolsShown: Bool { row.toolsShown }
    public var activity: String { row.activity }
    public var licenceURL: String, hfRepo: String, derivedDtypes: [String]
    /// The dtype the folder's weights are published in (fp32, bf16…); derived dtypes are made from it.
    public var sourceDtype: String
    public var licenceName: String?, licenceConditions: String?
    public var summary: String { row.summary }; public var publisher: String { row.publisher }; public var params: String { row.params }
    public var licence: String { row.licence }; public var commercial: Bool { row.commercial }; public var dtypes: [String] { row.precisions }
    public var downloadBytes: [String: Int64] { row.downloadBytes }; public var localCopyBytes: [String: Int64] { row.localCopyBytes }
    public var languages: [String], tasks: [String], modalitiesIn: [String], modalitiesOut: [String]
    /// The folder's own words: what the model is good for and what it can't do (empty when it declares neither).
    public var goodFor: [String] = [], limits: [String] = []
    public var suggested: Bool, popularity: JSONObject, sizesByDtype: [String: Int64], minRAMMB: Double, fitsRAM: Bool
    public var added: Bool, installed: Bool, loaded: Bool
    public var matchingDtype: String?
    public var formats: [ModelFormat] = []
    public var ramMBByDtype: [String: Double] = [:]
    public var fitsRAMByDtype: [String: Bool] = [:]
    public init(
        row: ModelRow, metadata: JSONObject = [:], sizesByDtype: [String: Int64]? = nil, added: Bool = false,
        memoryMB: Double = Double(ProcessInfo.processInfo.physicalMemory) / 1e6
    ) {
        self.row = row
        formats = row.formats
        description = metadata["description"]?.string ?? row.summary
        links = metadata["links"]?.object?.compactMapValues(\.string) ?? [:]
        runners = []; settingsSummary = []
        licenceName = metadata["licence"]?["name"]?.string; licenceConditions = metadata["licence"]?["conditions"]?.string
        licenceURL = metadata["licence"]?["url"]?.string ?? ""; hfRepo = metadata["weights"]?["repo"]?.string ?? ""
        if !hfRepo.isEmpty && links["huggingface"] == nil { links["huggingface"] = "https://huggingface.co/" + hfRepo }
        if !licenceURL.isEmpty && links["licence"] == nil { links["licence"] = licenceURL }
        let derived = metadata["derive"]?.object?.keys.sorted() ?? []
        derivedDtypes = derived
        sourceDtype = metadata["weights"]?["dtype"]?.string ?? row.precisions.first { !derived.contains($0) } ?? ""
        languages = metadata["languages"]?.array?.compactMap(\.string) ?? []
        tasks = metadata["tasks"]?.array?.compactMap(\.string) ?? []
        goodFor = metadata["good_for"]?.array?.compactMap(\.string) ?? []; limits = metadata["limits"]?.array?.compactMap(\.string) ?? []
        modalitiesIn = metadata["facets"]?["modalities_in"]?.array?.compactMap(\.string) ?? []
        modalitiesOut = metadata["facets"]?["modalities_out"]?.array?.compactMap(\.string) ?? []
        suggested = metadata["suggested"]?.bool ?? false; popularity = metadata["popularity"]?.object ?? [:]
        self.sizesByDtype = sizesByDtype ?? row.downloadBytes
        let dtypeSizes = self.sizesByDtype
        ramMBByDtype = Dictionary(
            uniqueKeysWithValues: row.precisions.map { dtype in
                (dtype, (dtype == row.selectedPrecision ? row.peakRAMMB : nil) ?? Double(dtypeSizes[dtype] ?? row.sizeBytes ?? 0) / 1e6 + 768)
            })
        fitsRAMByDtype = ramMBByDtype.mapValues { CatalogMatch.fits(ramMB: $0, physicalBytes: UInt64(max(0, memoryMB * 1e6))) }
        minRAMMB = ramMBByDtype[row.selectedPrecision] ?? ramMBByDtype.values.min() ?? 768
        fitsRAM = fitsRAMByDtype[row.selectedPrecision] ?? false
        matchingDtype = nil
        self.added = added; loaded = row.residentDType != nil; installed = loaded || row.states.values.contains { [.ready, .loading, .loaded].contains($0.state) }
    }
    /// The memory a precision needs, as Get more models, its fit mark and `find` show it: the figure Load admission uses
    /// (the table's recorded peak, then the folder's forecast) or an older recorded run, whichever is larger; then the
    /// declared format figure; the weight size plus 768 MB only when nothing was recorded.
    public static func ramMB(row: ModelRow, precision: String, forecast: PrecisionMemoryRecipe?, measuredMB: Double?, declaredMB: Double?, sizeBytes: Int64?) -> Double {
        let admitted = LoadMemoryRequirement(row: row, precision: precision, forecast: forecast).neededGB.map { $0 * 1000 }
        return [admitted, measuredMB].compactMap { $0 }.max() ?? declaredMB ?? Double(sizeBytes ?? row.sizeBytes ?? 0) / 1e6 + 768
    }
    /// After `ramMBByDtype` changed: each precision's fit mark, and the selected precision's need and fit as find and Get
    /// more models show them.
    public mutating func settleRAM(selected: String, physicalBytes: UInt64) {
        fitsRAMByDtype = ramMBByDtype.mapValues { CatalogMatch.fits(ramMB: $0, physicalBytes: physicalBytes) }
        minRAMMB = ramMBByDtype[selected] ?? ramMBByDtype.values.min() ?? minRAMMB
        fitsRAM = fitsRAMByDtype[selected] ?? fitsRAM
    }
    public func encode(to encoder: any Encoder) throws {
        var object = try WireJSON.decode(JSONEncoder().encode(row))
        object["row"] = .object(object)
        object["description"] = .string(description); object["links"] = .object(links.mapValues(JSONValue.string))
        object["runners"] = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(runners))
        object["settingsSummary"] = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(settingsSummary))
        object["licenceURL"] = .string(licenceURL); object["hfRepo"] = .string(hfRepo); object["derivedDtypes"] = .array(derivedDtypes.map(JSONValue.string))
        object["sourceDtype"] = .string(sourceDtype)
        if let licenceName { object["licenceName"] = .string(licenceName) }
        if let licenceConditions { object["licenceConditions"] = .string(licenceConditions) }
        object["fastAvailable"] = .bool(row.fastAvailable)
        object["dtypes"] = .array(dtypes.map(JSONValue.string))
        object["languages"] = .array(languages.map(JSONValue.string)); object["tasks"] = .array(tasks.map(JSONValue.string))
        object["goodFor"] = .array(goodFor.map(JSONValue.string)); object["limits"] = .array(limits.map(JSONValue.string))
        object["modalitiesIn"] = .array(modalitiesIn.map(JSONValue.string)); object["modalitiesOut"] = .array(modalitiesOut.map(JSONValue.string))
        object["suggested"] = .bool(suggested); object["popularity"] = .object(popularity)
        object["sizesByDtype"] = .object(sizesByDtype.mapValues { .number(Double($0)) })
        object["matchingDtype"] = matchingDtype.map(JSONValue.string) ?? .null
        object["formats"] = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(formats))
        object["ramMBByDtype"] = .object(ramMBByDtype.mapValues(JSONValue.number))
        object["fitsRAMByDtype"] = .object(fitsRAMByDtype.mapValues(JSONValue.bool))
        object["minRAMMB"] = .number(minRAMMB); object["fitsRAM"] = .bool(fitsRAM)
        object["added"] = .bool(added); object["installed"] = .bool(installed); object["loaded"] = .bool(loaded)
        try JSONValue.object(object).encode(to: encoder)
    }
    private func matching(_ query: CatalogQuery) -> Self? {
        // The conjunction belongs to ONE offered dtype, not independently to the smallest size and another dtype.
        let dtype = CatalogMatch.matchingDtype(self, query: query)
        // Diagnostic entries with no runnable dtype must remain visible in an unfiltered browser.
        guard let dtype else {
            return dtypes.isEmpty && query.dtypes.isEmpty && query.minSizeBytes == nil && query.maxSizeBytes == nil && query.minRAMMB == nil && query.maxRAMMB == nil
                && query.fitsRAM == nil ? self : nil
        }
        var entry = self; entry.matchingDtype = dtype; entry.row.selectedPrecision = dtype
        entry.row.sizeBytes = sizesByDtype[dtype]; entry.row.peakRAMMB = ramMBByDtype[dtype]
        if let reason = entry.row.dtypeUnavailableReason(dtype) {
            entry.row.verification = reason
            entry.row.states[dtype] = .init(state: .failed, error: reason, phase: "unsupportedFormat")
        }
        entry.minRAMMB = ramMBByDtype[dtype] ?? minRAMMB; entry.fitsRAM = fitsRAMByDtype[dtype] ?? fitsRAM
        return entry
    }
    public static func query(_ entries: [Self], _ query: CatalogQuery) -> [Self] {
        func matches(_ requested: [String], _ values: [String]) -> Bool { requested.isEmpty || !Set(requested).isDisjoint(with: values) }
        let filtered = entries.compactMap { original -> Self? in
            guard let e = original.matching(query) else { return nil }
            let text = [e.name, e.id, e.summary, e.publisher, e.params, e.kind, e.licence, e.hfRepo].joined(separator: " ")
            let included =
                (query.text.isEmpty || text.localizedStandardContains(query.text))
                && matches(query.kinds, [e.kind]) && matches(query.licences, [e.licence]) && matches(query.languages, e.languages)
                && matches(query.tasks, e.tasks) && matches(query.modalitiesIn, e.modalitiesIn) && matches(query.modalitiesOut, e.modalitiesOut)
                && (query.commercial == nil || query.commercial == e.commercial) && (query.suggested == nil || query.suggested == e.suggested)
                && (query.added == nil || query.added == e.added) && (query.installed == nil || query.installed == e.installed)
                && (query.loaded == nil || query.loaded == e.loaded)
            return included ? e : nil
        }
        func number(_ e: Self) -> Double? {
            switch query.sort {
            case .size: return Double(e.sizesByDtype[e.matchingDtype ?? e.row.selectedPrecision] ?? 0)
            case .ram: return e.minRAMMB
            case .popularity: return e.popularity["hf_downloads"]?.number ?? 0
            case .likes: return e.popularity["hf_likes"]?.number ?? 0
            case .released: return Double(e.row.released ?? 0)
            case .params:
                let text = e.params.uppercased(); let scale = text.hasSuffix("B") ? 1e9 : text.hasSuffix("M") ? 1e6 : text.hasSuffix("K") ? 1e3 : 1
                return (Double(text.trimmingCharacters(in: CharacterSet.letters)) ?? 0) * scale
            case .added: return e.added ? 1 : 0
            case .installed: return e.installed ? 1 : 0
            case .loaded: return e.loaded ? 1 : 0
            default: return nil
            }
        }
        func string(_ e: Self) -> String {
            switch query.sort {
            case .kind: return e.kind
            case .publisher: return e.publisher
            case .licence: return e.licence
            default: return e.name
            }
        }
        return filtered.sorted { a, b in
            if a.suggested != b.suggested { return a.suggested }
            let order: ComparisonResult
            if let x = number(a), let y = number(b) {
                order = x == y ? .orderedSame : x < y ? .orderedAscending : .orderedDescending
            } else {
                order = string(a).localizedStandardCompare(string(b))
            }
            if order == .orderedSame { return a.id < b.id }
            return query.descending ? order == .orderedDescending : order == .orderedAscending
        }
    }
}
public struct SettingState: Codable, Sendable, Equatable, Identifiable {
    public var declaration: JSONObject
    public var value: JSONValue
    public var pending: Bool
    public var id: String { declaration["id"]?.string ?? "" }
    public init(declaration: JSONObject, value: JSONValue, pending: Bool = false) { self.declaration = declaration; self.value = value; self.pending = pending }
}
public struct OperationPolicy: Codable, Sendable, Equatable {
    public var timeoutS: Double
    public var firstResponseTimeoutS: Double
    enum CodingKeys: String, CodingKey { case timeoutS = "timeout_s", firstResponseTimeoutS = "first_response_timeout_s" }
    public init(timeoutS: Double = 120, firstResponseTimeoutS: Double = 120) {
        self.timeoutS = timeoutS; self.firstResponseTimeoutS = firstResponseTimeoutS
    }
}
public struct ModelOperation: Codable, Sendable, Equatable, Identifiable {
    public var name: String, title: String, inputSchema: JSONObject, output: String, stream: Bool
    public var prepareImage: OperationImagePreparation?
    public var schemaVersion: Int?
    public var outputSchema: JSONObject?
    public var policy: OperationPolicy?
    public var annotations: JSONObject?
    public var id: String { name }
    public init(
        name: String, title: String, inputSchema: JSONObject, output: String, stream: Bool,
        outputSchema: JSONObject? = nil, policy: OperationPolicy? = nil, annotations: JSONObject? = nil, schemaVersion: Int? = nil
    ) {
        self.name = name; self.title = title; self.inputSchema = inputSchema; self.output = output; self.stream = stream
        self.schemaVersion = schemaVersion; self.outputSchema = outputSchema; self.policy = policy; self.annotations = annotations
    }
}
extension ModelsService {
    public func settingsPanel(id: String) async -> SettingsPanel? { nil }
    public func approveLocalCode(id: String, contentDigest: String, approvedBy: ApprovalSource) async throws { try await approveLocalCode(id: id, contentDigest: contentDigest) }
    public func saveDesignedVoice(id: String, previewID: String, name: String) async throws -> JSONObject {
        throw WorkerFailure(code: "unsupported", message: "This service does not support saving voice previews.")
    }
    public func deleteSavedVoice(id: String, voiceID: String) async throws { throw WorkerFailure(code: "unsupported", message: "This service does not support saved voices.") }
    public func setSettings(id: String, values: JSONObject) async throws { throw WorkerFailure(code: "unsupported", message: "This service does not support atomic settings.") }
    public func revokeLocalCode(id: String) async throws {
        throw WorkerFailure(code: "unsupported", message: "This service does not support local code consent.")
    }
    public func approveLocalCode(id: String, contentDigest: String) async throws {
        throw WorkerFailure(code: "unsupported", message: "This service does not support local code consent.")
    }
    public func approveLocalCode(id: String) async throws {
        guard let digest = await catalog(query: .init()).first(where: { $0.id == id })?.row.contentDigest else {
            throw WorkerFailure(code: "not_found", message: "Unknown local model \(id).")
        }
        try await approveLocalCode(id: id, contentDigest: digest)
    }
    public func operations(id: String) async -> [ModelOperation] { [] }
    public func catalog(query: CatalogQuery = .init()) async -> [CatalogEntry] { CatalogEntry.query(await rows().map { CatalogEntry(row: $0, added: true) }, query) }
    public func add(id: String) async throws { throw WorkerFailure(code: "unsupported", message: "This service does not support table membership.") }
    public func remove(id: String, deleteWeights: Bool = false) async throws {
        throw WorkerFailure(code: "unsupported", message: "This service does not support table membership.")
    }
    public func settings(id: String) async throws -> [SettingState] { [] }
    public func setSetting(id: String, key: String, value: JSONValue) async throws { throw WorkerFailure(code: "setting", message: "Unknown setting \(key).") }
    public func applySettings(id: String) async throws { try await unload(id: id); try await load(id: id) }
}
