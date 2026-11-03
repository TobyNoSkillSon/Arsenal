// SPDX-License-Identifier: MIT
import Foundation

/// A model's settings panel: `settings.toml` beside `model.toml` (docs/MODEL-FOLDER.md, "Settings panel"). It arranges
/// the folder's `[[setting]]` declarations into sections, picks each one's control, names choice values, and adds
/// presets (named bundles of values set together). It declares nothing new: every setting it names, and every value a
/// preset or condition uses, must be accepted by model.toml's declarations. One native component in the app draws
/// every panel; a folder without the file gets the generic list. Each kind's reference is `kinds/<kind>/settings.toml`,
/// with the declarations it expects in `kinds/<kind>/reference.toml`.
public struct SettingsPanel: Codable, Sendable, Equatable {
    public static let fileName = "settings.toml"
    public static let schemaVersion = 1
    /// Segments are equal cells in a fixed-width control: more than six do not fit with readable labels.
    public static let maxPresets = 6

    public var sections: [Section]
    /// Declared settings deliberately left out of the panel (still reachable through HTTP and MCP).
    public var hidden: [String]

    public struct Section: Codable, Sendable, Equatable {
        /// Only on a collapsed section: the disclosure row's label.
        public var title: String?
        public var collapsed: Bool
        public var showWhen: [String: [JSONValue]]
        public var rows: [Row]
    }
    public struct Row: Codable, Sendable, Equatable {
        public var setting: String?
        public var label: String?
        public var control: Control?
        /// Display names for values, keyed by `SettingsPanel.key(value)`.
        public var labels: [String: String]
        /// SF Symbol names for values (segments); the label or value is the segment's tooltip.
        public var icons: [String: String]
        public var unit: String?
        /// A dice button that picks a random value in the declaration's bounds (seeds).
        public var randomize: Bool
        public var showWhen: [String: [JSONValue]]
        public var presets: [Preset]
        public var isPresets: Bool { setting == nil }
    }
    public struct Preset: Codable, Sendable, Equatable {
        public var title: String
        public var icon: String?
        public var values: [String: JSONValue]
    }
    public enum Control: String, Codable, Sendable, CaseIterable {
        case `switch`, segments, menu, slider, field, editor, file, voice
    }

    /// The key a value is named by in `labels`, `icons` and error messages: `true`, `1024`, `0.5`, `english`.
    public static func key(_ value: JSONValue) -> String {
        switch value {
        case .bool(let b): return b ? "true" : "false"
        case .number(let n): return n == n.rounded() && abs(n) < 1e15 ? String(Int64(n)) : String(n)
        case .string(let s): return s
        default: return ""
        }
    }
    public static func key(_ scalar: ManifestScalar) -> String { key(json(scalar)) }
    public static func json(_ scalar: ManifestScalar) -> JSONValue {
        switch scalar {
        case .bool(let b): return .bool(b)
        case .number(let n): return .number(n)
        case .string(let s): return .string(s)
        }
    }

    /// Whether every condition holds for these current values (an absent condition holds).
    public static func holds(_ conditions: [String: [JSONValue]], values: [String: JSONValue]) -> Bool {
        conditions.allSatisfy { id, accepted in values[id].map { v in accepted.contains(v) } ?? false }
    }

    /// The folder's panel, nil when it has none. Throws one error listing every problem.
    public static func load(folder: URL, settings: [SettingDeclaration]) throws -> SettingsPanel? {
        let url = folder.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        _ = try SafePath.resolve(fileName, inside: folder)
        return try parse(String(contentsOf: url, encoding: .utf8), settings: settings)
    }

    /// The kinds with a reference panel in `kinds/<kind>/` (video is ready before its first model).
    public static let referenceKinds = ["images", "chat", "decisions", "voices", "sound", "enhancement", "video"]
    /// Every reference directory under `kinds/`: each kind's, plus variants for a different shape of the same settings
    /// (`images/resolution`: one size choice instead of free width and height; `voices/design`: a model that designs or
    /// clones voices and takes a language).
    public static let referenceDirectories = referenceKinds + ["images/resolution", "voices/design"]
    public static let referenceFileName = "reference.toml"
    struct ReferenceFile: Decodable { var setting: [SettingDeclaration] }
    /// A kind's reference: the declarations in `reference.toml` (each valid) and the panel arranging them.
    public static func reference(_ kindDirectory: URL) throws -> (settings: [SettingDeclaration], panel: SettingsPanel) {
        let text = try String(contentsOf: SafePath.resolve(referenceFileName, inside: kindDirectory), encoding: .utf8)
        let settings = try TOMLParser.decode(ReferenceFile.self, from: text).setting
        let problems = settings.flatMap { $0.problems() }
        guard problems.isEmpty else { throw ManifestError.invalid(problems.joined(separator: "; ")) }
        guard let panel = try load(folder: kindDirectory, settings: settings) else { throw ManifestError.invalid("\(kindDirectory.lastPathComponent): no \(fileName)") }
        return (settings, panel)
    }

    /// Where the service keeps a model's panel beside its folder metadata (and so in its catalog entry).
    public static let metadataKey = "settings_panel"
    /// A signed folder's panel from the catalog's file bytes (`files`: repository path → bytes), only when they
    /// match the signed size and hash and pass validation.
    public static func verified(_ item: CatalogModel, files: [String: Data]) -> JSONValue? {
        guard let file = item.files.first(where: { $0.path == fileName }), let data = files["models/\(item.id)/\(file.path)"],
            Int64(data.count) == file.bytes, FileDigest.sha256(data) == file.sha256,
            let panel = try? parse(String(decoding: data, as: UTF8.self), settings: item.manifest.settings)
        else { return nil }
        return panel.json
    }
    /// A local folder's panel, from its validated snapshot.
    public static func verified(_ local: LocalModelFolder) -> JSONValue? {
        guard local.fileDigests[fileName] != nil else { return nil }
        return (try? load(folder: local.directory, settings: local.manifest.settings))?.flatMap(\.json)
    }
    public var json: JSONValue? { (try? JSONEncoder().encode(self)).flatMap { try? JSONDecoder().decode(JSONValue.self, from: $0) } }
    public init(sections: [Section], hidden: [String] = []) { self.sections = sections; self.hidden = hidden }
    public init?(json: JSONValue?) {
        guard let json, let data = try? JSONEncoder().encode(json), let panel = try? JSONDecoder().decode(Self.self, from: data) else { return nil }
        self = panel
    }

    /// Validation for `ModelManifest.problems`: each problem as one line, prefixed with the file name.
    public static func folderProblems(folder: URL, settings: [SettingDeclaration]) -> [String] {
        do { _ = try load(folder: folder, settings: settings); return [] } catch {
            return error.localizedDescription.split(separator: "\n").map { "\(fileName): \($0)" }
        }
    }

    public static func parse(_ text: String, settings: [SettingDeclaration]) throws -> SettingsPanel {
        var parser = TOMLParser(text)
        let root: [String: TOMLValue]
        do { root = try parser.parse() } catch { throw ManifestError.invalid(error.localizedDescription) }
        var reader = Reader(declarations: Dictionary(settings.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a }))
        let panel = reader.panel(root)
        guard reader.errors.isEmpty, let panel else { throw ManifestError.invalid(reader.errors.joined(separator: "\n")) }
        return panel
    }

    /// Reads the parsed TOML into a panel, collecting every problem with where it is ("section 2, row 1").
    struct Reader {
        let declarations: [String: SettingDeclaration]
        var errors: [String] = []
        mutating func fail(_ at: String, _ text: String) { errors.append(at.isEmpty ? text : "\(at): \(text)") }

        mutating func unknownKeys(_ table: [String: TOMLValue], allowed: Set<String>, at: String) {
            for key in table.keys.sorted() where !allowed.contains(key) {
                fail(at, "unknown key \(key) (allowed: \(allowed.sorted().joined(separator: ", ")))")
            }
        }
        func scalar(_ value: TOMLValue) -> JSONValue? {
            switch value {
            case .string(let s): return .string(s)
            case .integer(let i): return .number(Double(i))
            case .decimal(let d): return .number(d)
            case .bool(let b): return .bool(b)
            default: return nil
            }
        }
        mutating func string(_ table: [String: TOMLValue], _ key: String, at: String) -> String? {
            guard let value = table[key] else { return nil }
            guard case .string(let s) = value, !s.trimmingCharacters(in: .whitespaces).isEmpty else { fail(at, "\(key) must be a nonempty string"); return nil }
            return s
        }
        mutating func bool(_ table: [String: TOMLValue], _ key: String, at: String) -> Bool {
            guard let value = table[key] else { return false }
            guard case .bool(let b) = value else { fail(at, "\(key) must be true or false"); return false }
            return b
        }
        mutating func tables(_ value: TOMLValue?, _ key: String, at: String) -> [[String: TOMLValue]] {
            guard let value else { return [] }
            guard case .array(let items) = value else { fail(at, "\(key) must be an array of tables"); return [] }
            return items.compactMap { item in
                guard case .table(let t) = item else { fail(at, "\(key) must be an array of tables"); return nil }
                return t
            }
        }
        /// A value as the declaration accepts it; nil (with an error) otherwise.
        mutating func accepted(_ value: TOMLValue, setting id: String, at: String) -> JSONValue? {
            guard let declaration = declarations[id] else { fail(at, "\(id) is not a setting declared in model.toml"); return nil }
            guard let json = scalar(value), let manifest = Self.manifestScalar(json), declaration.accepts(manifest) else {
                fail(at, "\(id) does not accept \(scalar(value).map { SettingsPanel.key($0) } ?? "this value")\(Self.allowed(declaration))")
                return nil
            }
            return json
        }
        static func manifestScalar(_ value: JSONValue) -> ManifestScalar? {
            switch value {
            case .bool(let b): return .bool(b)
            case .number(let n): return .number(n)
            case .string(let s): return .string(s)
            default: return nil
            }
        }
        static func allowed(_ d: SettingDeclaration) -> String {
            switch d.type {
            case "choice": return " (choices: \((d.choices ?? []).map { SettingsPanel.key($0) }.joined(separator: ", ")))"
            case "toggle": return " (true or false)"
            case "number", "slider":
                let bounds = [
                    d.min.map { "min \(SettingsPanel.key(JSONValue.number($0)))" }, d.max.map { "max \(SettingsPanel.key(JSONValue.number($0)))" },
                    d.step.map { "step \(SettingsPanel.key(JSONValue.number($0)))" }
                ]
                let text = bounds.compactMap { $0 }.joined(separator: ", ")
                return text.isEmpty ? " (a number)" : " (\(text))"
            default: return " (a string)"
            }
        }
        /// The keys a setting's values go by: choices, true/false; nil for free values.
        func valueKeys(_ d: SettingDeclaration) -> Set<String>? {
            switch d.type {
            case "choice": return Set((d.choices ?? []).map { SettingsPanel.key($0) })
            case "toggle": return ["true", "false"]
            default: return nil
            }
        }
        mutating func conditions(_ value: TOMLValue?, at: String) -> [String: [JSONValue]] {
            guard let value else { return [:] }
            guard case .table(let table) = value, !table.isEmpty else { fail(at, "show_when must be a table of setting = value or [values]"); return [:] }
            var result: [String: [JSONValue]] = [:]
            for (id, raw) in table.sorted(by: { $0.key < $1.key }) {
                let values: [TOMLValue]
                if case .array(let list) = raw { values = list } else { values = [raw] }
                if values.isEmpty { fail(at, "show_when.\(id) needs at least one value") }
                let accepted = values.compactMap { accepted($0, setting: id, at: at + " show_when") }
                if accepted.count == values.count { result[id] = accepted }
            }
            return result
        }
        /// The control a row may use for its setting type; the first is the default.
        static func controls(_ d: SettingDeclaration) -> [Control] {
            switch d.type {
            case "toggle": return [.switch, .segments]
            case "choice": return [.segments, .menu]
            case "slider": return [.slider, .field]
            case "number": return d.min != nil && d.max != nil ? [.field, .slider] : [.field]
            case "voice": return [.voice]
            default: return [.field, .editor, .file]
            }
        }

        mutating func panel(_ root: [String: TOMLValue]) -> SettingsPanel? {
            unknownKeys(root, allowed: ["schema", "section", "hidden"], at: "")
            if root["schema"] != .integer(Int64(SettingsPanel.schemaVersion)) { fail("", "schema = \(SettingsPanel.schemaVersion) is required") }
            var hidden: [String] = []
            if let value = root["hidden"] {
                if case .array(let list) = value,
                    list.allSatisfy({
                        if case .string = $0 { return true }; return false
                    })
                {
                    hidden = list.compactMap {
                        if case .string(let s) = $0 { return s }; return nil
                    }
                } else {
                    fail("", "hidden must be an array of setting ids")
                }
            }
            for id in hidden where declarations[id] == nil { fail("hidden", "\(id) is not a setting declared in model.toml") }
            if Set(hidden).count != hidden.count { fail("hidden", "lists a setting twice") }
            let rawSections = tables(root["section"], "section", at: "")
            if rawSections.isEmpty { fail("", "at least one [[section]] is required") }
            var sections: [Section] = []
            for (s, table) in rawSections.enumerated() {
                let at = "section \(s + 1)"
                unknownKeys(table, allowed: ["title", "collapsed", "show_when", "row"], at: at)
                let collapsed = bool(table, "collapsed", at: at)
                let title = string(table, "title", at: at)
                if collapsed && title == nil { fail(at, "a collapsed section needs a title (its disclosure row's label)") }
                if !collapsed && title != nil { fail(at, "title is only for a collapsed section (panels have no headings)") }
                let showWhen = conditions(table["show_when"], at: at)
                let rawRows = tables(table["row"], "row", at: at)
                if rawRows.isEmpty { fail(at, "needs at least one [[section.row]]") }
                var rows: [Row] = []
                for (r, row) in rawRows.enumerated() { if let read = self.row(row, at: "\(at), row \(r + 1)") { rows.append(read) } }
                sections.append(Section(title: title, collapsed: collapsed, showWhen: showWhen, rows: rows))
            }
            // Coverage: each declared setting has exactly one row, or is hidden on purpose.
            let rowSettings = sections.flatMap(\.rows).compactMap(\.setting)
            for id in Set(rowSettings) where rowSettings.filter({ $0 == id }).count > 1 { fail("", "\(id) has more than one row") }
            for id in hidden where rowSettings.contains(id) { fail("hidden", "\(id) also has a row") }
            for id in declarations.keys.sorted() where !rowSettings.contains(id) && !hidden.contains(id) {
                fail("", "\(id) is declared in model.toml but has no row; add one or list it in hidden")
            }
            for (s, section) in sections.enumerated() {
                for (r, row) in section.rows.enumerated() where row.isPresets {
                    for id in Set(row.presets.flatMap(\.values.keys)).sorted() where !rowSettings.contains(id) && !hidden.contains(id) && declarations[id] != nil {
                        fail("section \(s + 1), row \(r + 1)", "preset sets \(id), which needs its own row so a value that matches no preset shows")
                    }
                }
            }
            return SettingsPanel(sections: sections, hidden: hidden)
        }

        mutating func row(_ table: [String: TOMLValue], at: String) -> Row? {
            unknownKeys(table, allowed: ["setting", "label", "control", "labels", "icons", "unit", "randomize", "show_when", "presets"], at: at)
            let label = string(table, "label", at: at)
            let showWhen = conditions(table["show_when"], at: at)
            if table["setting"] != nil && table["presets"] != nil { fail(at, "a row has either setting or presets, not both"); return nil }
            if table["presets"] != nil {
                for key in ["control", "labels", "icons", "unit", "randomize"] where table[key] != nil { fail(at, "\(key) is for setting rows, not preset rows") }
                if label == nil { fail(at, "a preset row needs a label") }
                let presets = self.presets(table["presets"], at: at)
                return Row(setting: nil, label: label, control: .segments, labels: [:], icons: [:], unit: nil, randomize: false, showWhen: showWhen, presets: presets)
            }
            guard let id = string(table, "setting", at: at) else { fail(at, "a row needs setting = \"<id>\" or presets = [...]"); return nil }
            guard let declaration = declarations[id] else { fail(at, "\(id) is not a setting declared in model.toml"); return nil }
            var control: Control?
            if let raw = string(table, "control", at: at) {
                if let c = Control(rawValue: raw), Self.controls(declaration).contains(c) {
                    control = c
                } else {
                    fail(at, "control \(raw) does not fit a \(declaration.type) setting (use \(Self.controls(declaration).map(\.rawValue).joined(separator: " or ")))")
                }
            }
            let labels = names(table["labels"], "labels", declaration: declaration, at: at)
            let icons = names(table["icons"], "icons", declaration: declaration, at: at)
            if !icons.isEmpty, let control, control != .segments { fail(at, "icons are drawn on segments; drop control or use segments") }
            if !icons.isEmpty, control == nil, !Self.controls(declaration).contains(.segments) { fail(at, "icons need a choice or toggle setting") }
            let unit = string(table, "unit", at: at)
            if unit != nil, !["number", "slider"].contains(declaration.type) { fail(at, "unit is for number and slider settings") }
            let randomize = bool(table, "randomize", at: at)
            if randomize, !["number", "slider"].contains(declaration.type) || declaration.min == nil || declaration.max == nil {
                fail(at, "randomize needs a number or slider setting with min and max")
            }
            if control == .segments || (control == nil && !icons.isEmpty), declaration.type == "choice", (declaration.choices ?? []).count > SettingsPanel.maxPresets {
                fail(at, "\(id) has more than \(SettingsPanel.maxPresets) choices; use control = \"menu\"")
            }
            return Row(
                setting: id, label: label, control: control ?? (icons.isEmpty ? nil : .segments), labels: labels, icons: icons, unit: unit, randomize: randomize,
                showWhen: showWhen, presets: [])
        }

        /// `labels` / `icons`: value key → nonempty string, for values the setting can take.
        mutating func names(_ value: TOMLValue?, _ key: String, declaration: SettingDeclaration, at: String) -> [String: String] {
            guard let value else { return [:] }
            guard case .table(let table) = value else { fail(at, "\(key) must be a table of value = \"text\""); return [:] }
            guard let keys = valueKeys(declaration) else { fail(at, "\(key) are for choice and toggle settings"); return [:] }
            var result: [String: String] = [:]
            for (k, v) in table {
                guard keys.contains(k) else { fail(at, "\(key).\(k) is not a value of \(declaration.id) (values: \(keys.sorted().joined(separator: ", ")))"); continue }
                guard case .string(let s) = v, !s.isEmpty else { fail(at, "\(key).\(k) must be a nonempty string"); continue }
                result[k] = s
            }
            return result
        }

        mutating func presets(_ value: TOMLValue?, at: String) -> [Preset] {
            let raw = tables(value, "presets", at: at)
            guard (2...SettingsPanel.maxPresets).contains(raw.count) else { fail(at, "presets needs 2 to \(SettingsPanel.maxPresets) entries"); return [] }
            var presets: [Preset] = []
            for (i, table) in raw.enumerated() {
                let pat = "\(at), preset \(i + 1)"
                unknownKeys(table, allowed: ["title", "icon", "values"], at: pat)
                let title = string(table, "title", at: pat)
                if title == nil { fail(pat, "needs a title (the segment's label, or its tooltip when it has an icon)") }
                let icon = string(table, "icon", at: pat)
                guard case .table(let rawValues)? = table["values"], !rawValues.isEmpty else { fail(pat, "values must be a nonempty table of setting = value"); continue }
                var values: [String: JSONValue] = [:]
                for (id, v) in rawValues { if let accepted = accepted(v, setting: id, at: pat) { values[id] = accepted } }
                if values.count == rawValues.count, let title { presets.append(Preset(title: title, icon: icon, values: values)) }
            }
            if Set(presets.map(\.title)).count != presets.count { fail(at, "preset titles must be unique") }
            if let first = presets.first, presets.contains(where: { Set($0.values.keys) != Set(first.values.keys) }) {
                fail(at, "every preset in a row sets the same settings (so exactly one can match)")
            }
            for (i, a) in presets.enumerated() where presets[..<i].contains(where: { $0.values == a.values }) { fail(at, "presets \(a.title) repeats another preset's values") }
            if presets.contains(where: { $0.icon != nil }), presets.contains(where: { $0.icon == nil }) { fail(at, "give every preset an icon, or none") }
            return presets
        }
    }
}
