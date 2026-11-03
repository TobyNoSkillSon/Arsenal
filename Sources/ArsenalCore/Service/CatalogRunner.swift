// SPDX-License-Identifier: MIT
import Foundation

public struct RunnerHistory: Codable, Sendable, Equatable {
    public var path: String, title: String
    public var verdict: String?
    public init(path: String, title: String, verdict: String? = nil) { self.path = path; self.title = title; self.verdict = verdict }
}
public struct CatalogRunner: Codable, Sendable, Equatable, Identifiable {
    public var name: String, language: String, engine: String
    public var acceptedDefault: Bool, eligible: Bool, selected: Bool
    public var stats: [String: PrecisionStatSet] = [:]
    public var features: [RunnerFeature] = []
    public var results: [JSONObject]
    public var history: RunnerHistory?
    /// Why an ineligible runner cannot run on this Mac ("Needs macOS 27", "Built for M4"); nil when eligible.
    public var ineligibleReason: String?
    public var id: String { name }
    public init(
        name: String, language: String, engine: String, acceptedDefault: Bool = false, eligible: Bool = true, selected: Bool = false, results: [JSONObject] = [],
        history: RunnerHistory? = nil, ineligibleReason: String? = nil
    ) {
        self.name = name; self.language = language; self.engine = engine; self.acceptedDefault = acceptedDefault; self.eligible = eligible; self.selected = selected
        self.results = results; self.history = history
        self.ineligibleReason = ineligibleReason
    }
    private enum CodingKeys: String, CodingKey {
        case stats, features, name, language, engine, acceptedDefault, eligible, selected, results, history, ineligibleReason
    }
    /// Payloads from before v2.1's readiness fields decode with their defaults.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            name: try c.decode(String.self, forKey: .name), language: try c.decode(String.self, forKey: .language), engine: try c.decode(String.self, forKey: .engine),
            acceptedDefault: try c.decodeIfPresent(Bool.self, forKey: .acceptedDefault) ?? false, eligible: try c.decodeIfPresent(Bool.self, forKey: .eligible) ?? true,
            selected: try c.decodeIfPresent(Bool.self, forKey: .selected) ?? false, results: try c.decodeIfPresent([JSONObject].self, forKey: .results) ?? [],
            history: try c.decodeIfPresent(RunnerHistory.self, forKey: .history),
            ineligibleReason: try c.decodeIfPresent(String.self, forKey: .ineligibleReason))
        stats = try c.decodeIfPresent([String: PrecisionStatSet].self, forKey: .stats) ?? [:]
        features = try c.decodeIfPresent([RunnerFeature].self, forKey: .features) ?? []
    }
}

/// History bytes are verified when loading the catalog, not read by a SwiftUI row.
enum CatalogHistory {
    static func latest(_ files: [String: Data], runner: String) -> RunnerHistory? {
        let paths = files.keys.filter { $0.hasPrefix("log/") && $0.hasSuffix(".md") }.sorted { $0.localizedStandardCompare($1) == .orderedDescending }
        for path in paths {
            guard let data = files[path], data.count <= 1_048_576 else { continue }
            let text = String(decoding: data, as: UTF8.self)
            let fence = text.hasPrefix("+++\n") ? "+++" : text.hasPrefix("---\n") ? "---" : nil
            let parts = fence.map { text.components(separatedBy: $0) } ?? [text]
            var header: JSONObject = [:]
            if parts.count >= 3 {
                if fence == "+++" {
                    header = (try? TOMLParser.decode(JSONObject.self, from: parts[1])) ?? [:]
                } else {
                    // Only scalar fields used for the pointer; the catalog validator owns full YAML admission.
                    for line in parts[1].split(separator: "\n") {
                        let pair = line.split(separator: ":", maxSplits: 1)
                        guard pair.count == 2 else { continue }
                        let key = pair[0].trimmingCharacters(in: .whitespaces)
                        if ["runner", "title", "verdict"].contains(key) {
                            let value = pair[1].trimmingCharacters(in: .whitespaces)
                            let decoded =
                                (try? JSONDecoder().decode(String.self, from: Data(value.utf8)))
                                ?? (value.hasPrefix("'") && value.hasSuffix("'") ? String(value.dropFirst().dropLast()) : value)
                            header[key] = .string(decoded)
                        }
                    }
                }
            }
            // Older folder histories describe the model as a whole. A runner-specific entry never labels another runner.
            if let declared = header["runner"]?.string, declared != runner { continue }
            let body = parts.count >= 3 ? parts.dropFirst(2).joined(separator: fence ?? "") : text
            let heading = body.split(separator: "\n").first { $0.hasPrefix("# ") }.map { String($0.dropFirst(2)) }
            let title = header["title"]?.string ?? heading ?? URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
            return RunnerHistory(path: path, title: title, verdict: header["verdict"]?.string)
        }
        return nil
    }
}
extension ModelsService {
    public func setRunner(id: String, runner: String?) async throws { throw WorkerFailure(code: "unsupported", message: "This service does not support runner choices.") }
}
