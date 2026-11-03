// SPDX-License-Identifier: MIT
import Foundation

/// Shared by real and stub services so pending flags and defaults cannot drift.
struct ModelSettings: Codable {
    /// Saved values, normalised: they change only through `load` (disk reads), `set` and `remove`, so no path can bring
    /// back a value saved under a retired key.
    private(set) var values: [String: JSONObject] = [:]
    var applied: [String: JSONObject] = [:]
    var declarations: [String: [JSONObject]] = [:]
    /// Retired keys still accepted (`[[setting_alias]]`): alias id → value key → the settings it stands for.
    var aliases: [String: [String: [String: JSONObject]]] = [:]
    /// The app's placement switch is not a runner setting. Split and validate it before any
    /// settings/config mutation, so a combined settings write commits once or changes nothing.
    static func splitMenuPlacement(_ values: JSONObject) throws -> (settings: JSONObject, shown: Bool?) {
        var settings = values
        let placement = settings.removeValue(forKey: "__tools")
        if let placement, placement.bool == nil { throw WorkerFailure(code: "invalid_setting", message: "__tools must be a boolean.") }
        return (settings, placement?.bool)
    }
    init() {}
    enum CodingKeys: String, CodingKey { case values, applied, declarations, aliases }
    /// Recovery files written before aliases existed have no `aliases`: they decode with none, values normalised.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        values = try c.decodeIfPresent([String: JSONObject].self, forKey: .values) ?? [:]
        applied = try c.decodeIfPresent([String: JSONObject].self, forKey: .applied) ?? [:]
        declarations = try c.decodeIfPresent([String: [JSONObject]].self, forKey: .declarations) ?? [:]
        aliases = try c.decodeIfPresent([String: [String: [String: JSONObject]]].self, forKey: .aliases) ?? [:]
        for id in aliases.keys { normalize(id) }
    }
    /// A model's declarations and aliases from its folder metadata; its saved values are normalised against them.
    mutating func declare(_ id: String, metadata: JSONObject?) {
        declarations[id] = metadata?["setting"]?.array?.compactMap(\.object) ?? []
        var table: [String: [String: JSONObject]] = [:]
        for alias in metadata?["setting_alias"]?.array?.compactMap(\.object) ?? [] {
            guard let name = alias["id"]?.string else { continue }
            table[name] = alias["values"]?.object?.compactMapValues(\.object) ?? [:]
        }
        aliases[id] = table.isEmpty ? nil : table
        normalize(id)
    }
    /// The one way saved settings enter (launch, Refresh, a merged config write): every model's aliases are normalised,
    /// so no path sees a value saved under a retired key.
    mutating func load(_ saved: [String: JSONObject]) {
        values = saved
        for id in aliases.keys { normalize(id) }
    }
    /// A value saved under an alias moves to the settings it stands for, one target at a time: a value saved for a
    /// target wins, a missing one is filled from the alias (a saved FLUX `resolution` fills width and height). The
    /// alias key goes; the next settings write persists the result.
    /// Forget all saved values and declarations for a model removed from the table.
    mutating func forget(_ id: String) {
        values[id] = nil; applied[id] = nil; declarations[id] = nil; aliases[id] = nil
    }
    /// Forget a saved value (the setting falls back to its default).
    mutating func remove(_ id: String, key: String) { values[id]?[key] = nil }
    /// A request to a loaded model whose settings snapshot (`self`) may be older than the live ones. Request-time
    /// settings never need a reload, so they come from the live declarations and values, the unset meaning included (a
    /// seed with no value sends none); load-time settings stay with the snapshot until Reload and are not part of a
    /// request. A retired request key that only the snapshot declares (FLUX's `resolution`, before width and height)
    /// is added as the live alias value equal to the current settings, when one is; the live keys still carry the size.
    func residentRequest(_ id: String, params: JSONObject, live: ModelSettings, schema: JSONObject? = nil) -> JSONObject {
        guard let current = live.declarations[id] else { return request(id, params: params, schema: schema) }
        var result = live.request(id, params: params, schema: schema)
        let names = Set(current.compactMap { $0["id"]?.string })
        let effective = Dictionary(live.states(id, loaded: false).map { ($0.id, $0.value) }, uniquingKeysWith: { a, _ in a })
        for declaration in declarations[id] ?? [] where declaration["applies"]?.string == "request" {
            guard let name = declaration["id"]?.string, !names.contains(name), let table = live.aliases[id]?[name],
                let mapping = declaration["maps_to"]?.string, mapping.hasPrefix("request.")
            else { continue }
            let path = String(mapping.dropFirst(8))
            guard result[String(path.split(separator: ".").first ?? "")] == nil, schema.map({ OperationSchemaValidator.declares(path: path, in: $0) }) ?? true,
                let key = table.first(where: { $0.value.allSatisfy { effective[$0.key] == $0.value } })?.key
            else { continue }
            Self.insert(declaration["choices"]?.array?.first { SettingsPanel.key($0) == key } ?? .string(key), path: path, into: &result, overriding: false)
        }
        return result
    }
    mutating func normalize(_ id: String) {
        for (name, mapping) in aliases[id] ?? [:] {
            guard let saved = values[id]?[name] else { continue }
            values[id]?[name] = nil
            for (target, value) in mapping[SettingsPanel.key(saved)] ?? [:] where values[id]?[target] == nil { values[id, default: [:]][target] = value }
        }
    }
    func states(_ id: String, loaded: Bool) -> [SettingState] {
        (declarations[id] ?? []).map { declaration in
            let key = declaration["id"]?.string ?? ""
            let saved = values[id]?[key]
            let value = saved.flatMap { Self.accepts($0, declaration: declaration) ? $0 : nil } ?? declaration["default"] ?? .null
            return SettingState(declaration: declaration, value: value, pending: loaded && declaration["applies"]?.string == "load" && applied[id]?[key] != value)
        }
    }
    func loadValues(_ id: String) -> JSONObject {
        Dictionary(uniqueKeysWithValues: states(id, loaded: false).filter { $0.declaration["applies"]?.string == "load" }.map { ($0.id, $0.value) })
    }
    mutating func set(_ id: String, key: String, value: JSONValue) throws {
        if declarations[id]?.contains(where: { $0["id"]?.string == key }) != true, let mapping = aliases[id]?[key] {
            guard let targets = mapping[SettingsPanel.key(value)] else {
                throw WorkerFailure(code: "setting", message: "\(key): use one of \(mapping.keys.sorted().joined(separator: ", ")).")
            }
            for (target, value) in targets.sorted(by: { $0.key < $1.key }) { try set(id, key: target, value: value) }
            return
        }
        guard let declaration = declarations[id]?.first(where: { $0["id"]?.string == key }) else { throw WorkerFailure(code: "setting", message: "Unknown setting \(key).") }
        // A setting without a default (a seed) has no value until one is pinned; null clears it back to the runner's choice.
        if value == .null, declaration["default"] == nil { values[id]?[key] = nil; return }
        guard Self.accepts(value, declaration: declaration) else { throw WorkerFailure(code: "setting", message: "\(key): value is outside its current declaration.") }
        values[id, default: [:]][key] = value
    }
    private static func accepts(_ value: JSONValue, declaration: JSONObject) -> Bool {
        var declaration = declaration; if declaration["title"] == nil { declaration["title"] = declaration["id"] }
        guard let data = try? WireJSON.encode(declaration), let rule = try? JSONDecoder().decode(SettingDeclaration.self, from: data),
            let encoded = try? JSONEncoder().encode(value), let scalar = try? JSONDecoder().decode(ManifestScalar.self, from: encoded)
        else { return false }
        return rule.accepts(scalar)
    }
    func invalidSavedKeys(_ id: String) -> [String] {
        (declarations[id] ?? []).compactMap { declaration in
            guard let key = declaration["id"]?.string, let saved = values[id]?[key], !Self.accepts(saved, declaration: declaration) else { return nil }; return key
        }
    }
    func warning(_ id: String) -> String? {
        let keys = invalidSavedKeys(id)
        return keys.isEmpty ? nil : "Saved \(keys.joined(separator: ", ")) no longer valid; using the model default."
    }
    mutating func discardInvalidSavedValues() { for id in declarations.keys { for key in invalidSavedKeys(id) { values[id]?[key] = nil } } }
    func loadConfig(_ id: String, base: JSONObject) -> JSONObject {
        var config = base; config["settings"] = .object(loadValues(id))
        for state in states(id, loaded: false) where state.declaration["applies"]?.string == "load" {
            if let mapping = state.declaration["maps_to"]?.string, mapping.hasPrefix("config.") {
                Self.insert(state.value, path: String(mapping.dropFirst(7)), into: &config, overriding: true)
            }
        }
        return config
    }
    func request(_ id: String, params: JSONObject, schema: JSONObject? = nil) -> JSONObject {
        var result = params
        for state in states(id, loaded: false) where state.declaration["applies"]?.string == "request" {
            if state.value == .null || state.value == .string("") || state.value == .object([:]) { continue }
            if let mapping = state.declaration["maps_to"]?.string, mapping.hasPrefix("request.") {
                let path = String(mapping.dropFirst(8))
                let group = String(path.split(separator: ".").first ?? "")
                // Caller intent owns the entire group: never splice a saved transcript into a caller's clip.
                let conflicts = state.declaration["excludes"]?.array?.compactMap(\.string) ?? (group == "reference" ? ["voice"] : [])
                if params[group] != nil || conflicts.contains(where: { params[$0] != nil }) { continue }
                if let schema, !OperationSchemaValidator.declares(path: path, in: schema) { continue }
                // An op that works on reference images keeps their size unless it requires one: a saved width or
                // height would override "Same as reference" in every edit.
                if let schema, ["width", "height"].contains(path), schema["properties"]?["images"] != nil,
                    !(schema["required"]?.array?.contains(.string(path)) ?? false)
                {
                    continue
                }
                // An op may have a more useful default than the shared setting (music vs SFX).
                // Valid saved settings and explicit caller fields still take precedence.
                let saved = values[id]?[state.id].map { Self.accepts($0, declaration: state.declaration) } == true
                let opDefault = schema.flatMap { Self.property(path: path, in: $0)?["default"] }
                Self.insert(saved ? state.value : opDefault ?? state.value, path: path, into: &result, overriding: false)
            }
        }
        return result
    }
    private static func property(path: String, in schema: JSONObject) -> JSONObject? {
        var rule = schema
        for part in path.split(separator: ".") {
            guard let child = rule["properties"]?.object?[String(part)]?.object else { return nil }
            rule = child
        }
        return rule
    }
    private static func insert(_ value: JSONValue, path: String, into object: inout JSONObject, overriding: Bool) {
        let parts = path.split(separator: ".").map(String.init)
        guard let first = parts.first else { return }
        if parts.count == 1 { if overriding || object[first] == nil { object[first] = value }; return }
        // A scalar supplied by the caller overrides the entire subtree, including defaults.
        if !overriding, let current = object[first], current.object == nil { return }
        var child = object[first]?.object ?? [:]
        insert(value, path: parts.dropFirst().joined(separator: "."), into: &child, overriding: overriding)
        object[first] = .object(child)
    }
}
