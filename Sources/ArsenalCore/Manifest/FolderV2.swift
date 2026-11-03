// SPDX-License-Identifier: MIT
import Foundation

public struct PopularitySnapshot: Codable, Sendable, Equatable {
    public var hfDownloads: Int64; public var hfLikes: Int64; public var asOf: String
    enum CodingKeys: String, CodingKey { case hfDownloads = "hf_downloads", hfLikes = "hf_likes", asOf = "as_of" }
    public init(hfDownloads: Int64, hfLikes: Int64, asOf: String) { self.hfDownloads = hfDownloads; self.hfLikes = hfLikes; self.asOf = asOf }
}
public struct ModelFacets: Codable, Sendable, Equatable {
    public var modalitiesIn: [String]; public var modalitiesOut: [String]
    enum CodingKeys: String, CodingKey { case modalitiesIn = "modalities_in", modalitiesOut = "modalities_out" }
    public init(modalitiesIn: [String], modalitiesOut: [String]) { self.modalitiesIn = modalitiesIn; self.modalitiesOut = modalitiesOut }
    public func problems() -> [String] {
        let allowed = Set(["text", "audio", "image", "video"])
        return [modalitiesIn, modalitiesOut].allSatisfy { !$0.isEmpty && Set($0).isSubset(of: allowed) && Set($0).count == $0.count }
            ? [] : ["facets require unique, nonempty text/audio/image/video modalities"]
    }
}
public struct SettingDeclaration: Codable, Sendable, Equatable, Identifiable {
    public var id: String; public var title: String; public var type: String
    /// Absent only on a request-time number setting: no value is sent, so the runner picks one (a seed: random each
    /// request).
    public var defaultValue: ManifestScalar?; public var choices: [ManifestScalar]?
    public var min: Double?; public var max: Double?; public var step: Double?
    public var applies: String; public var mapsTo: String?; public var help: String?
    public var excludes: [String]?
    enum CodingKeys: String, CodingKey { case id, title, type, choices, min, max, step, applies, help, excludes; case defaultValue = "default", mapsTo = "maps_to" }
    public init(
        id: String, title: String, type: String, defaultValue: ManifestScalar?, choices: [ManifestScalar]? = nil, min: Double? = nil, max: Double? = nil, step: Double? = nil,
        applies: String, mapsTo: String? = nil, help: String? = nil
    ) {
        self.id = id; self.title = title; self.type = type; self.defaultValue = defaultValue; self.choices = choices; self.min = min; self.max = max; self.step = step
        self.applies = applies; self.mapsTo = mapsTo; self.help = help
    }
    public func accepts(_ value: ManifestScalar) -> Bool {
        switch (type, value) {
        case ("toggle", .bool): return true
        case ("text", .string), ("voice", .string): return true
        case ("choice", _): return choices?.contains(value) == true
        case ("number", .number(let n)), ("slider", .number(let n)):
            guard n.isFinite, min.map({ n >= $0 }) ?? true, max.map({ n <= $0 }) ?? true else { return false }
            if let step, step > 0 { let units = (n - (min ?? 0)) / step; return abs(units - units.rounded()) < 1e-7 }
            return true
        default: return false
        }
    }
    public func problems() -> [String] {
        var errors: [String] = []
        if id.range(of: "^[a-z][a-z0-9_]*$", options: .regularExpression) == nil || title.isEmpty { errors.append("invalid setting id/title") }
        if !["load", "request"].contains(applies) || (applies == "request" && mapsTo == nil)
            || (mapsTo.map { !$0.hasPrefix(applies == "load" ? "config." : "request.") || $0.split(separator: ".").count < 2 } ?? false)
        {
            errors.append("setting \(id) has invalid applies/maps_to")
        }
        if let mapsTo, applies == "load", ["kernels", "assets", "settings"].contains(String(mapsTo.dropFirst(7).split(separator: ".").first ?? "")) {
            errors.append("setting \(id) targets a reserved load key")
        }
        if let excludes, excludes.contains(where: { $0.range(of: "^[a-z][a-z0-9_]*$", options: .regularExpression) == nil }) { errors.append("setting \(id) has invalid excludes") }
        if !["toggle", "choice", "slider", "number", "text", "voice"].contains(type) { errors.append("setting \(id) has unknown type") }
        if type == "choice", (choices?.isEmpty ?? true) || Set((choices ?? []).map { String(describing: $0) }).count != choices?.count {
            errors.append("setting \(id) needs unique choices")
        }
        if [min, max, step].compactMap({ $0 }).contains(where: { !$0.isFinite }) || (step.map { $0 <= 0 } ?? false) || (min != nil && max != nil && min! > max!) {
            errors.append("setting \(id) has invalid bounds/step")
        }
        if type == "slider" && (min == nil || max == nil || step == nil) { errors.append("slider \(id) requires min/max/step") }
        if let defaultValue {
            if !accepts(defaultValue) { errors.append("setting \(id) default is outside its declaration") }
        } else if type != "number" || applies != "request" {
            // A load setting always reaches the runner's config, so it needs a value; a request setting can stay unsent.
            errors.append("setting \(id) needs a default (only a request-time number setting may leave it out, meaning the runner picks)")
        }
        return errors
    }
}
/// A retired setting key kept working (`[[setting_alias]]`): each of its values stands for values of current settings.
/// A settings write of the alias sets those settings, and a value saved under it moves to them when they hold none.
/// FLUX's `resolution = "512x512"` became `width = 512, height = 512`.
public struct SettingAlias: Codable, Sendable, Equatable {
    public var id: String
    public var values: [String: [String: ManifestScalar]]
    public init(id: String, values: [String: [String: ManifestScalar]]) { self.id = id; self.values = values }
    public func problems(settings: [SettingDeclaration]) -> [String] {
        var errors: [String] = []
        if id.range(of: "^[a-z][a-z0-9_]*$", options: .regularExpression) == nil { errors.append("invalid setting_alias id \(id)") }
        if settings.contains(where: { $0.id == id }) { errors.append("setting_alias \(id) is also a declared setting") }
        if values.isEmpty { errors.append("setting_alias \(id) needs values") }
        for (key, targets) in values.sorted(by: { $0.key < $1.key }) {
            if targets.isEmpty { errors.append("setting_alias \(id).\(key) sets nothing") }
            for (target, value) in targets.sorted(by: { $0.key < $1.key }) {
                guard let declaration = settings.first(where: { $0.id == target }) else {
                    errors.append("setting_alias \(id).\(key) names \(target), which is not a declared setting"); continue
                }
                if !declaration.accepts(value) { errors.append("setting_alias \(id).\(key): \(target) does not accept that value") }
            }
        }
        return errors
    }
}
public struct OpDeclaration: Codable, Sendable, Equatable {
    public var name: String; public var title: String; public var input: String; public var output: String; public var stream: Bool
    public var prepareImage: OperationImagePreparation?
    public var outputSchema: String?
    public var timeoutS: Double?
    public var firstResponseTimeoutS: Double?
    public var annotations: JSONObject?
    public var defaultOp: Bool?
    public var isDefault: Bool { defaultOp ?? false }
    enum CodingKeys: String, CodingKey {
        case prepareImage = "prepare_image"
        case name, title, input, output, stream
        case outputSchema = "output_schema", defaultOp = "default", timeoutS = "timeout_s", firstResponseTimeoutS = "first_response_timeout_s", annotations
    }
    public init(name: String, title: String, input: String, output: String, stream: Bool, outputSchema: String? = nil, defaultOp: Bool? = nil) {
        self.defaultOp = defaultOp; self.name = name; self.title = title; self.input = input; self.output = output; self.stream = stream; self.outputSchema = outputSchema
    }
    public func problems(folder: URL, kindsDirectory: URL? = nil) -> [String] {
        var errors: [String] = []
        if name.range(of: "^[a-z][a-z0-9_]*$", options: .regularExpression) == nil || title.isEmpty || !["json", "audio", "image", "text-stream", "file"].contains(output) {
            errors.append("invalid op \(name) name/title/output")
        }
        if let prepareImage {
            if !ModelManifest.validID(prepareImage.model) || prepareImage.operation.range(of: "^[a-z][a-z0-9_]*$", options: .regularExpression) == nil
                || prepareImage.fileID.isEmpty || prepareImage.fileID.count > 128
            {
                errors.append("op \(name) has invalid prepare_image model/operation/file_id")
            }
        }
        for bound in [timeoutS, firstResponseTimeoutS].compactMap({ $0 }) where !bound.isFinite || bound < 1 || bound > 3600 {
            errors.append("op \(name) timeout must be 1...3600 seconds")
        }
        for path in [input, outputSchema].compactMap({ $0 }) {
            do { _ = try OperationSchema.read(path: path, folder: folder, kindsDirectory: kindsDirectory) } catch { errors.append("op \(name): \(error.localizedDescription)") }
        }
        return errors
    }
}
public struct RunnerSelection: Codable, Sendable, Equatable {
    public var defaultRunner: String; public var chips: [String: String]?
    enum CodingKeys: String, CodingKey { case defaultRunner = "default", chips }
    public init(defaultRunner: String, chips: [String: String] = [:]) { self.defaultRunner = defaultRunner; self.chips = chips }
    public func runner(chip: String) -> String { chips?[Self.family(chip)] ?? defaultRunner }
    public static func family(_ chip: String) -> String { chip.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).first.map(String.init) ?? chip.lowercased() }
}
public enum OperationSchema {
    /// Paths are local to the model, or explicitly rooted under kinds/. No remote references or code execute.
    public static func read(path: String, folder: URL, kindsDirectory: URL? = nil) throws -> Data {
        let url: URL
        if path.hasPrefix("kinds/"), let kindsDirectory {
            url = try SafePath.resolve(String(path.dropFirst(6)), inside: kindsDirectory)
        } else {
            url = try SafePath.resolve(path, inside: folder)
        }
        let data = legacyKeywordsRenamed(try Data(contentsOf: url))
        guard data.count <= 1024 * 1024, let schema = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            schema["$schema"] as? String == "https://json-schema.org/draft/2020-12/schema"
        else { throw ManifestError.invalid("\(path) must be a draft 2020-12 JSON Schema object (max 1 MiB)") }
        try ModelSchemaValidation.checkSchema(schemaData: data)
        return data
    }
    /// Schemas in folders written before the rename (user-added local models, cached catalog kinds) annotate with the
    /// legacy `x-…` prefix; they read as the current `x-arsenal-` keywords. Only quoted keys change.
    static func legacyKeywordsRenamed(_ data: Data) -> Data {
        let legacy = Data(#""x-mobius-"#.utf8) // legacy-compat: schema keywords written before the rename
        guard data.range(of: legacy) != nil else { return data }
        var result = Data(), rest = data[...]
        while let range = rest.range(of: legacy) {
            result.append(rest[..<range.lowerBound]); result.append(Data(#""x-arsenal-"#.utf8)); rest = rest[range.upperBound...]
        }
        result.append(rest)
        return result
    }
}

/// A single installed-model operation prepares the request's image. Its selected PNG must be an RGBA cutout.
public struct OperationImagePreparation: Codable, Sendable, Equatable {
    public var model: String
    public var operation: String
    public var fileID: String
    public init(model: String, operation: String, fileID: String) { self.model = model; self.operation = operation; self.fileID = fileID }
    enum CodingKeys: String, CodingKey { case model, operation; case fileID = "file_id" }
}
