// SPDX-License-Identifier: MIT
import Foundation

public struct ModelManifest: Codable, Sendable, Equatable {
    private static func sharedKinds(near folder: URL) -> URL {
        let root = folder.deletingLastPathComponent().deletingLastPathComponent()
        let resources = root.appendingPathComponent("Resources/kinds")
        return FileManager.default.fileExists(atPath: resources.path) ? resources : root.appendingPathComponent("kinds")
    }

    public var precision: NativeHalfPolicy?
    public var architecture: String?
    public var quantization: QuantizationDeclaration?
    public var schema: Int
    public var id: String
    public var name: String
    public var kind: String
    public var kindFile: String?
    public var description: String?
    public var links: [String: String]?
    public var summary: String
    /// What the model does in a few plain words ("Background removal"), shown under its name; see ModelPurpose.
    public var purpose: String?
    public var goodFor: [String]?
    public var limits: [String]?
    public var publisher: String
    public var released: Int
    public var tools: [String]?
    public var activity: String?
    public var activitySection: String { activity ?? tools?.first ?? "default" }
    public var tasks: [String]
    public var languages: [String]
    public var params: String
    public var context: Int?
    public var nativeContext: Int?
    public var scaffold: Bool?
    public var suggested: Bool?
    public var popularity: PopularitySnapshot?
    public var facets: ModelFacets?
    public var setting: [SettingDeclaration]?
    public var settingAlias: [SettingAlias]?
    public var op: [OpDeclaration]?
    public var selection: RunnerSelection?
    public var licence: ModelLicence
    public var weights: WeightSpec
    public var memory: PrecisionMemoryRecipe?
    public var derive: [String: String]?
    public var vendor: [String: WeightSpec]?
    public var format: [ModelFormat]?
    public var asset: [AssetSpec]?
    public var assets: [String]?
    public var runner: [RunnerSpec]
    public var menu: ModelMenu?
    public var gate: RecipeSpec
    public var bench: RecipeSpec
    enum CodingKeys: String, CodingKey {
        case precision, architecture, quantization, tools, activity, description, links, schema, id, name, kind, summary, purpose, publisher, released, tasks, languages, params,
            context
        case scaffold, licence, weights, memory, derive, vendor, format, asset, assets, runner, menu, gate, bench, suggested, popularity, facets, setting, op, selection
        case goodFor = "good_for"
        case limits
        case nativeContext = "native_context", kindFile = "kind_file", settingAlias = "setting_alias"
    }
    public var runners: [RunnerSpec] { runner }
    public var basePrecision: String {
        if normalizesNativeHalf { return precision?.dtype ?? "fp16" }
        if derive?["fp16"]?.hasPrefix("cast:") == true { return "fp16" }
        if derive?["bf16"]?.hasPrefix("cast:") == true { return "bf16" }
        return weights.dtype == "fp32" ? "fp16" : weights.dtype
    }
    public var precisions: [String] {
        let extra = Set((derive ?? [:]).keys).union((vendor ?? [:]).keys).union((format ?? []).map(\.id)).subtracting([basePrecision])
        return [basePrecision] + ["int8", "int4"].filter { extra.contains($0) } + extra.subtracting(["int8", "int4"]).sorted()
    }
    public static func load(_ folder: URL, requirePins: Bool = true, kinds: Set<String>? = nil, kindsDirectory: URL? = nil) throws -> Self {
        let file = try SafePath.resolve("model.toml", inside: folder)
        let model = try TOMLParser.decode(Self.self, from: String(contentsOf: file, encoding: .utf8))
        let problems = model.problems(folder: folder, requirePins: requirePins, kinds: kinds, kindsDirectory: kindsDirectory)
        guard problems.isEmpty else { throw ManifestError.invalid(problems.joined(separator: "; ")) }; return model
    }
    public func problems(folder: URL, requirePins: Bool = true, kinds: Set<String>? = nil, kindsDirectory: URL? = nil) -> [String] {
        var errors: [String] = []
        func need(_ ok: Bool, _ message: String) { if !ok { errors.append("\(id): \(message)") } }
        need(scaffold != true, "unfinished metadata scaffold; fill the README checklist and remove scaffold = true")
        need([1, 2, 3].contains(schema), "schema must be 1, 2 or 3")
        if let tools {
            need(
                Set(tools).isSubset(of: ToolNames.all) && Set(tools).count == tools.count,
                "tools must be unique names from: " + ToolNames.all.sorted().joined(separator: ", "))
            if let op {
                let declared = Set(op.map(\.name))
                for tool in tools {
                    guard let offered = ToolNames.capability[tool] else { continue }
                    need(!offered.isDisjoint(with: declared), "tools names \(tool), but no declared operation is one it offers (\(offered.sorted().joined(separator: ", ")))")
                }
            }
        }
        if let purpose {
            need(
                !purpose.trimmingCharacters(in: .whitespaces).isEmpty && purpose.count <= ModelPurpose.maxLength && !purpose.contains(where: \.isNewline),
                "purpose must be one short line of at most \(ModelPurpose.maxLength) characters")
        }
        for (key, values) in [("good_for", goodFor), ("limits", limits)] {
            if let values {
                need(
                    values.count <= 32 && values.allSatisfy { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.count <= 1000 },
                    "\(key) must contain up to 32 nonempty short entries")
            }
        }
        if let activity { need(activity.range(of: "^[a-z][a-z0-9_-]*$", options: .regularExpression) != nil, "activity must be a section identifier") }
        if let architecture {
            need(
                !architecture.isEmpty && architecture.count <= 128 && architecture.range(of: "^[A-Za-z0-9_][A-Za-z0-9_.-]*$", options: .regularExpression) != nil,
                "invalid architecture selector")
        }
        if let quantization {
            need(
                quantization.eligibleMatrices == 0 && !quantization.reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && quantization.reason.count <= 1000,
                "quantization declares zero eligible matrices with a reason")
            need(derive?["int8"] != nil && derive?["int4"] != nil, "zero-matrix models still declare int8 and int4 recipes")
        }
        if let precision {
            need(
                runners.contains { NativeHalfPolicy.normalizedWorkers.contains($0.worker ?? "") },
                "[precision] is only for the normalized native families: " + NativeHalfPolicy.normalizedWorkers.joined(separator: ", "))
            need(["fp16", "bf16"].contains(precision.dtype), "precision.dtype must be fp16 or bf16")
            errors += rawSourceProblems(precision, folder: folder)
            if precision.dtype == "bf16" {
                need(precision.reason?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false, "BF16 exception needs precision.reason")
                if let evidence = precision.evidence, SafePath.isRelative(evidence) {
                    need((try? SafePath.resolve(evidence, inside: folder)).map { FileManager.default.fileExists(atPath: $0.path) } == true, "BF16 exception evidence is missing")
                } else {
                    need(false, "BF16 exception needs a relative precision.evidence file")
                }
            }
        }
        if normalizesNativeHalf {
            need(derive?[basePrecision] == nil || derive?[basePrecision]?.hasPrefix("cast:") == true, "native half conversion recipe must start with cast:")
            need(
                derive?.keys.contains(where: { ["fp16", "bf16"].contains($0) && $0 != basePrecision }) != true,
                "native family offers only its declared serving half; declare BF16 with evidence")
        }
        need(Self.validID(id), "invalid model id")
        need(folder.lastPathComponent == id, "folder name must equal id")
        for (label, text) in [("name", name), ("summary", summary), ("publisher", publisher), ("params", params), ("kind", kind), ("licence.spdx", licence.spdx)] {
            need(!text.isEmpty, "\(label) is required")
        }
        need(!tasks.isEmpty && !languages.isEmpty, "tasks and languages must not be empty")
        if let nativeContext { need(nativeContext > 0, "native_context must be a positive token count") }
        if let kindFile {
            need(schema == 3 && kind.hasPrefix("local.") && Self.validID(kind), "embedded kinds require schema 3 and local.* ID")
            need(
                kinds?.contains(kind) != true || EmbeddedKind.repeatsSigned(model: self, folder: folder, kindsDirectory: kindsDirectory ?? Self.sharedKinds(near: folder)),
                "embedded kind cannot override a signed kind")
            do { _ = try EmbeddedKind.resolve(model: self, folder: folder) } catch { errors.append(error.localizedDescription) }
            need(SafePath.isRelative(kindFile), "unsafe kind_file")
        } else if let kinds {
            need(kinds.contains(kind), "unknown kind \(kind)")
        }
        need(URL(string: licence.url)?.scheme == "https", "licence.url must be HTTPS")
        need(weights.repo.split(separator: "/").count == 2 && weights.repo.split(separator: "/").allSatisfy { Self.validRepoPart(String($0)) }, "invalid Hugging Face repo")
        need(Self.isHex(weights.revision, count: 40), "weights.revision must be a 40-hex commit")
        need(
            ["fp16", "bf16"].contains(weights.dtype)
                || (weights.dtype == "fp32" && (normalizesNativeHalf || derive?["fp16"] != nil || derive?["bf16"]?.hasPrefix("cast:") == true)),
            "fp32 source requires a cast to a selected half format")
        need(!weights.files.isEmpty && Set(weights.files).count == weights.files.count, "weights.files must be a nonempty exact unique list")
        for file in weights.files {
            need(SafePath.isRelative(file), "unsafe weight path \(file)")
            if requirePins {
                need(Self.isHex(weights.sha256?[file] ?? "", count: 64), "missing sha256 for \(file)"); need((weights.bytes?[file] ?? -1) >= 0, "missing bytes for \(file)")
            }
        }
        for key in (weights.sha256 ?? [:]).keys { need(weights.files.contains(key), "unlisted sha256 \(key)") }
        for key in (weights.bytes ?? [:]).keys { need(weights.files.contains(key), "unlisted size \(key)") }
        for (precision, recipe) in derive ?? [:] {
            need(
                Self.validDtype(precision) && !recipe.isEmpty && (precision != weights.dtype || ["fp16", "bf16"].contains(precision) && recipe.hasPrefix("cast:")),
                "unsupported derive dtype or empty recipe \(precision)")
            need(vendor?[precision] == nil, "precision is both vendor and derived: \(precision)")
        }
        for (precision, source) in vendor ?? [:] {
            need(Self.validDtype(precision) && precision != basePrecision, "invalid vendor dtype")
            need(source.dtype == precision, "vendor dtype must match its label")
            checkSource(source, label: "vendor.\(precision)", requirePins: requirePins, errors: &errors)
        }
        for operation in operations where operation.prepareImage != nil {
            need(schema == 3 && operation.prepareImage?.model != id, "prepare_image requires schema 3 and another model")
        }
        let assetSpecs = asset ?? []
        need(Set(assetSpecs.map(\.id)).count == assetSpecs.count, "duplicate asset id")
        for spec in assetSpecs {
            need(Self.validID(spec.id) && ["model", "dataset"].contains(spec.repoType), "invalid asset id/repo_type")
            checkSource(spec.weights, label: "asset.\(spec.id)", requirePins: requirePins, errors: &errors)
            if let into = spec.into {
                need(SafePath.isRelative(into), "asset.\(spec.id).into must be a safe relative directory")
                need(derive?[basePrecision]?.hasPrefix("cast:") == true, "asset.\(spec.id).into needs the model's own cast: recipe for its base precision")
            }
        }
        let intoDirectories = assetSpecs.compactMap(\.into)
        need(Set(intoDirectories).count == intoDirectories.count, "asset into directories must be unique")
        for id in assets ?? [] { need(assetSpecs.contains { $0.id == id }, "asset declaration missing for \(id)") }
        need(!runner.isEmpty && Set(runner.map(\.name)).count == runner.count, "runners must be nonempty and unique")
        for r in runner {
            need(Self.validID(r.name), "unsafe runner name")
            need(!r.language.isEmpty && r.proto == 1, "unsupported runner language/protocol")
            need(!r.chips.isEmpty && r.minMacOS.split(separator: ".").allSatisfy { Int($0) != nil }, "invalid runner chips/min_macos")
            need(
                r.language != "python" && r.python == nil && r.lock == nil && r.envDownloadBytes == nil,
                "Python model runners are unsupported; use a native runner and Python only as a reference")
            if r.language == "swift", r.entry != nil {
                need(schema == 3, "folder-local Swift entry requires schema 3")
                do { try LocalSwiftRunner.validate(r, folder: folder) } catch { errors.append("\(id): \(error.localizedDescription)") }
            } else {
                need(r.worker != nil && Self.validID(r.worker ?? ""), "native runner needs worker")
            }
        }
        for path in [gate.recipe, bench.recipe, "README.md"] {
            do { let p = try SafePath.resolve(path, inside: folder); need(FileManager.default.fileExists(atPath: p.path), "missing \(path)") } catch {
                errors.append("\(id): \(error.localizedDescription)")
            }
        }
        let logs = (try? FileManager.default.contentsOfDirectory(atPath: folder.appendingPathComponent("log").path)) ?? []
        need(logs.contains { $0.hasSuffix(".md") && !$0.hasPrefix(".") }, "at least one research log entry required")
        do { _ = try SafePath.files(in: folder) } catch { errors.append("\(id): \(error.localizedDescription)") }
        let actions = (menu?.actions ?? []) + (menu?.sections ?? []).flatMap(\.items)
        for section in menu?.sections ?? [] { need(!section.title.isEmpty && !section.items.isEmpty, "menu section needs title and items") }
        need(Set(actions.map(\.id)).count == actions.count, "duplicate menu action")
        for action in actions { need(Self.validID(action.id) && !action.op.isEmpty && ["clipboard", "none"].contains(action.input), "invalid menu action") }
        if let memory { errors += memory.problems.map { "\(id): " + $0 } }
        errors += formatProblems()
        errors += v2Problems(folder: folder, kindsDirectory: kindsDirectory)
        errors += (settingAlias ?? []).flatMap { $0.problems(settings: settings) }.map { "\(id): \($0)" }
        errors += SettingsPanel.folderProblems(folder: folder, settings: settings).map { "\(id): \($0)" }
        return errors
    }
    public var resolvedTools: [String] { tools ?? (schema == 3 ? ["operation"] : []) }
    public var settings: [SettingDeclaration] { setting ?? [] }
    public var operations: [OpDeclaration] { op ?? [] }
    public var isSuggested: Bool { suggested ?? false }
    public func resolvedOperations(kind: KindSpec?) -> [OpDeclaration] {
        if let op { return op.sorted { $0.name < $1.name } }
        return (kind?.operations ?? []).filter(\.isDefault).sorted { $0.name < $1.name }
    }
    private func v2Problems(folder: URL, kindsDirectory: URL?) -> [String] {
        var errors: [String] = []
        if schema >= 2 && selection == nil { errors.append("\(id): schema 2 requires selection") }
        if let selection {
            for name in [selection.defaultRunner] + Array((selection.chips ?? [:]).values) where !runner.contains(where: { $0.name == name }) {
                errors.append("\(id): selection references unknown runner \(name)")
            }
            for chip in (selection.chips ?? [:]).keys where chip.range(of: "^m[0-9]+$", options: .regularExpression) == nil {
                errors.append("\(id): invalid selection chip \(chip)")
            }
        }
        if let popularity, popularity.hfDownloads < 0 || popularity.hfLikes < 0 || popularity.asOf.range(of: "^[0-9]{4}-[0-9]{2}-[0-9]{2}$", options: .regularExpression) == nil {
            errors.append("\(id): invalid popularity snapshot")
        }
        errors += (facets?.problems() ?? []).map { "\(id): " + $0 }
        if Set(settings.map(\.id)).count != settings.count { errors.append("\(id): duplicate setting") }
        errors += settings.flatMap { $0.problems() }.map { "\(id): " + $0 }
        if Set(operations.map(\.name)).count != operations.count { errors.append("\(id): duplicate op") }
        let kinds =
            kindFile != nil
            ? folder.appendingPathComponent("kinds") : kindsDirectory ?? Self.sharedKinds(near: folder)
        errors += operations.flatMap { $0.problems(folder: folder, kindsDirectory: kinds) }.map { "\(id): " + $0 }
        let kindURL = kinds.appendingPathComponent(kind + ".toml")
        let standard = (try? String(contentsOf: kindURL, encoding: .utf8)).flatMap { try? TOMLParser.decode(KindSpec.self, from: $0) }
        if schema == 3 {
            let resolved = resolvedOperations(kind: standard)
            if resolved.isEmpty { errors.append("schema 3 requires operations") }
            if resolved.contains(where: { $0.outputSchema == nil }) { errors.append("schema 3 requires every output_schema") }
        }
        do { try RunnerCheckSamples.validate(folder: folder, model: self, kind: standard, kindsDirectory: kinds) } catch { errors.append("\(id): \(error.localizedDescription)") }
        do { _ = try KernelManifest.loadIfPresent(folder: folder) } catch { errors.append("\(id): \(error.localizedDescription)") }
        return errors
    }
    private func checkSource(_ source: WeightSpec, label: String, requirePins: Bool, errors: inout [String]) {
        if !Self.isHex(source.revision, count: 40) || source.repo.split(separator: "/").count != 2 || !SafePath.isRelative(source.repo) || source.files.isEmpty
            || Set(source.files).count != source.files.count
        {
            errors.append("\(id): invalid \(label) source")
        }
        for file in source.files {
            if !SafePath.isRelative(file) { errors.append("\(id): unsafe \(label) file \(file)") }
            if requirePins && (!Self.isHex(source.sha256?[file] ?? "", count: 64) || (source.bytes?[file] ?? -1) < 0) { errors.append("\(id): missing \(label) pin for \(file)") }
        }
    }
    public static func validID(_ text: String) -> Bool { text.range(of: "^[a-z0-9][a-z0-9.-]*$", options: .regularExpression) != nil }
    public static func isHex(_ text: String, count: Int) -> Bool { text.count == count && text.allSatisfy { $0.isASCII && $0.isHexDigit } }
    private static func validRepoPart(_ text: String) -> Bool { ![".", ".."].contains(text) && text.range(of: "^[A-Za-z0-9][A-Za-z0-9_.-]*$", options: .regularExpression) != nil }
}
/// `name`: display name for a `LicenseRef-*` id; `conditions`: what limits commercial use when it is not plainly allowed.
public struct ModelLicence: Codable, Sendable, Equatable {
    public var spdx: String; public var commercial: Bool; public var url: String; public var name: String?; public var conditions: String?
}
public struct WeightSpec: Codable, Sendable, Equatable {
    public var repo: String; public var revision: String; public var dtype: String; public var files: [String]; public var sha256: [String: String]?
    public var bytes: [String: Int64]?; public var format: String?; public var group: Int?; public var repoType: String?
    enum CodingKeys: String, CodingKey { case repo, revision, dtype, files, sha256, bytes, format, group; case repoType = "repo_type" }
}
public struct RecipeSpec: Codable, Sendable, Equatable { public var recipe: String }
public struct RunnerSpec: Codable, Sendable, Equatable {
    public var name: String; public var language: String; public var worker: String?; public var proto: Int; public var chips: [String]; public var minMacOS: String
    public var fastExact: Bool
    public var python: String?; public var entry: String?; public var lock: String?
    /// Legacy Python declarations are decoded only to reject unsupported model folders explicitly.
    public var envDownloadBytes: Int64?
    public var features: [RunnerFeature]?
    enum CodingKeys: String, CodingKey {
        case features, name, language, worker, chips, python, entry, lock
        case proto = "protocol", minMacOS = "min_macos", fastExact = "fast_exact", envDownloadBytes = "env_download_bytes"
    }
}
public struct MenuSpec: Codable, Sendable, Equatable {
    public var id: String; public var title: String; public var op: String; public var input: String; public var params: [String: ManifestScalar]?
}
public enum ManifestScalar: Codable, Sendable, Equatable {
    case string(String), number(Double), bool(Bool)
    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(Bool.self) { self = .bool(v) } else if let v = try? c.decode(Double.self) { self = .number(v) } else { self = .string(try c.decode(String.self)) }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        }
    }
}

extension ModelManifest {
    /// The assets this model uses (`assets` selects; absent means every declared asset).
    public var selectedAssets: [AssetSpec] { (asset ?? []).filter { assets == nil || assets!.contains($0.id) } }
    /// Selected assets converted into the base snapshot at Get (`into`); they have no snapshot of their own.
    public var incorporatedAssets: [AssetSpec] { selectedAssets.filter { $0.into != nil } }
    /// Selected assets kept as their own verified snapshots and passed to Load.
    public var separateAssets: [AssetSpec] { selectedAssets.filter { $0.into == nil } }
}

public struct AssetSpec: Codable, Sendable, Equatable {
    public var id: String; public var repo: String; public var repoType: String; public var revision: String; public var files: [String]
    public var licence: AssetLicence; public var sha256: [String: String]?; public var bytes: [String: Int64]?
    /// The base snapshot subdirectory the model's own Get conversion writes this asset into (in the serving half
    /// format; derived tiers pack it with the rest). Such an asset is staged only while the base is converted and is
    /// never kept as a separate snapshot; Load reads `weights/<into>`. Absent: a separate verified snapshot, as before.
    public var into: String?
    enum CodingKeys: String, CodingKey { case id, repo, revision, files, licence, sha256, bytes, into; case repoType = "repo_type" }
    public var weights: WeightSpec { WeightSpec(repo: repo, revision: revision, dtype: "asset", files: files, sha256: sha256, bytes: bytes, repoType: repoType) }
}
public enum AssetLicence: Codable, Sendable, Equatable {
    case spdx(String), detail(ModelLicence)
    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer(); if let s = try? c.decode(String.self) { self = .spdx(s) } else { self = .detail(try c.decode(ModelLicence.self)) }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .spdx(let s): try c.encode(s)
        case .detail(let v): try c.encode(v)
        }
    }
}

/// A deliberate no-compression tier, never an implicit exemption from precision checks.
public struct QuantizationDeclaration: Codable, Sendable, Equatable {
    public var eligibleMatrices: Int
    public var reason: String
    public init(eligibleMatrices: Int, reason: String) { self.eligibleMatrices = eligibleMatrices; self.reason = reason }
    enum CodingKeys: String, CodingKey { case eligibleMatrices = "eligible_matrices", reason }
}
extension ModelManifest {
    public var precisionDtypes: [String: String] {
        guard quantization?.eligibleMatrices == 0 else { return [:] }
        return Dictionary(uniqueKeysWithValues: ["int8", "int4"].filter { precisions.contains($0) }.map { ($0, basePrecision) })
    }
    public var precisionNotes: [String: String] {
        var notes: [String: String] = [:]
        if normalizesNativeHalf { notes[basePrecision] = "Runs as " + basePrecision.uppercased() + "; normalized file by file at Get." }
        guard let quantization, quantization.eligibleMatrices == 0 else { return notes }
        let note = "Runs as \(basePrecision.uppercased()); no eligible Linear/Embedding matrices and no weight compression. " + quantization.reason
        for dtype in ["int8", "int4"] where precisions.contains(dtype) { notes[dtype] = note }
        return notes
    }
}
