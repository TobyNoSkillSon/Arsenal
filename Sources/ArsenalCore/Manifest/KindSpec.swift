// SPDX-License-Identifier: MIT
import Foundation

public struct KindColumn: Codable, Sendable, Equatable {
    public var id: String
    public var title: String
    public var unit: String
    public init(id: String, title: String, unit: String = "") { self.id = id; self.title = title; self.unit = unit }
}
public struct KindSpec: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var title: String
    public var order: Int
    public var columns: [KindColumn]
    public var api: [String: String]
    public var icon: String?
    public var titleSingular: String?
    public var op: [OpDeclaration]?
    public var facets: ModelFacets?
    public var primaryOperation: String?
    public var comparisonMetric: String?
    public var comparisonDirection: String?
    public var operations: [OpDeclaration] { op ?? [] }
    enum CodingKeys: String, CodingKey {
        case id, title, order, columns, api, suite, icon, op, facets
        case titleSingular = "title_singular", primaryOperation = "primary_operation", comparisonMetric = "comparison_metric", comparisonDirection = "comparison_direction"
    }
    public var suite: String
    public init(
        id: String, title: String, order: Int, columns: [KindColumn], api: [String: String] = [:], suite: String = "", icon: String? = nil, titleSingular: String? = nil,
        operations: [OpDeclaration] = [], facets: ModelFacets? = nil
    ) {
        self.id = id; self.title = title; self.order = order; self.columns = columns; self.api = api; self.suite = suite; self.icon = icon; self.titleSingular = titleSingular
        self.op = operations; self.facets = facets
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id); title = try c.decode(String.self, forKey: .title)
        order = try c.decodeIfPresent(Int.self, forKey: .order) ?? 100
        columns = try c.decodeIfPresent([KindColumn].self, forKey: .columns) ?? []
        api = try c.decodeIfPresent([String: String].self, forKey: .api) ?? [:]
        suite = try c.decodeIfPresent(String.self, forKey: .suite) ?? ""
        icon = try c.decodeIfPresent(String.self, forKey: .icon); titleSingular = try c.decodeIfPresent(String.self, forKey: .titleSingular)
        op = try c.decodeIfPresent([OpDeclaration].self, forKey: .op); facets = try c.decodeIfPresent(ModelFacets.self, forKey: .facets)
        primaryOperation = try c.decodeIfPresent(String.self, forKey: .primaryOperation)
        comparisonMetric = try c.decodeIfPresent(String.self, forKey: .comparisonMetric)
        comparisonDirection = try c.decodeIfPresent(String.self, forKey: .comparisonDirection)
    }
    public static func load(directory: URL) throws -> [Self] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil).filter { $0.pathExtension == "toml" }
        return try files.map { file in
            _ = try SafePath.resolve(file.lastPathComponent, inside: directory)
            let spec = try TOMLParser.decode(Self.self, from: String(contentsOf: file, encoding: .utf8))
            guard ModelManifest.validID(spec.id), file.deletingPathExtension().lastPathComponent == spec.id, !spec.title.isEmpty, !spec.suite.isEmpty,
                Set(spec.columns.map(\.id)).count == spec.columns.count
            else { throw ManifestError.invalid("invalid kind file \(file.lastPathComponent)") }
            if spec.op != nil || spec.facets != nil {
                guard !(spec.icon?.isEmpty ?? true), !(spec.titleSingular?.isEmpty ?? true) else { throw ManifestError.invalid("v2 kind needs icon and title_singular") }
            }
            guard Set(spec.operations.map(\.name)).count == spec.operations.count else { throw ManifestError.invalid("duplicate kind operation") }
            let errors = spec.operations.flatMap { $0.problems(folder: directory.appendingPathComponent(spec.id), kindsDirectory: directory) } + (spec.facets?.problems() ?? [])
            guard errors.isEmpty else { throw ManifestError.invalid(errors.joined(separator: "; ")) }
            return spec
        }.sorted { ($0.order, $0.id) < ($1.order, $1.id) }
    }
}
