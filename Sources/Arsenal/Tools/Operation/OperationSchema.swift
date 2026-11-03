// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// The generic operation tool's form, read from an operation's input schema (the bounded draft-2020-12 subset the service
// validates with). Each top-level property becomes one control: text, number, switch, choice, a file well for a property
// annotated `x-arsenal-file`, and an editable JSON field for whatever a plain control cannot express faithfully (objects,
// non-file arrays, unions). The form never decides validity: the service's own validator does, before and on submit.

/// One operation as the service resolved it (from the API's operation descriptors).
struct OperationDescriptor: Equatable, Identifiable {
    var name: String
    var title: String
    var inputSchema: JSONObject
    var outputSchema: JSONObject?
    /// "json", "file", "image", "audio" or "text-stream".
    var output: String
    var stream: Bool
    var annotations: JSONObject = [:]
    /// Seconds the service allows the whole operation (its declared policy); nil: the app's default.
    var timeout: Double?
    var id: String { name }
    /// Results are files the service hands back through job-scoped URLs (JSON negotiation).
    var returnsFiles: Bool { ["file", "image", "audio"].contains(output) }
}

extension OperationDescriptor {
    /// The service's resolved operation as the API describes it.
    init(_ op: ModelOperation) {
        self.init(
            name: op.name, title: op.title, inputSchema: op.inputSchema, outputSchema: op.outputSchema, output: op.output, stream: op.stream,
            annotations: op.annotations ?? [:], timeout: op.policy?.timeoutS)
    }
}

/// A file a field accepts: `{"accept": ["image/png", "image/jpeg"]}`.
struct FileAccept: Equatable {
    var mimes: [String]
    /// Accepts this file, judged by its extension's type (no file is opened to sniff it).
    func accepts(_ url: URL) -> Bool {
        guard !mimes.isEmpty else { return true }
        let mime = FileAccept.mime(url)
        return mimes.contains { pattern in
            pattern == "*/*" || pattern == mime || (pattern.hasSuffix("/*") && mime?.hasPrefix(String(pattern.dropLast())) == true)
        }
    }
    static func mime(_ url: URL) -> String? {
        switch url.pathExtension.lowercased() {
        case "png": return "image/png"
        case "jpg", "jpeg": return "image/jpeg"
        case "webp": return "image/webp"
        case "gif": return "image/gif"
        case "tif", "tiff": return "image/tiff"
        case "heic": return "image/heic"
        case "wav": return "audio/wav"
        case "flac": return "audio/flac"
        case "mp3": return "audio/mpeg"
        case "m4a": return "audio/mp4"
        case "txt": return "text/plain"
        case "json": return "application/json"
        case "pdf": return "application/pdf"
        case "aif", "aiff": return "audio/aiff"
        case "aac": return "audio/aac"
        case "ogg", "oga": return "audio/ogg"
        case "opus": return "audio/opus"
        case "caf": return "audio/x-caf"
        case "mp4", "m4v": return "video/mp4"
        case "mov": return "video/quicktime"
        case "webm": return "video/webm"
        case "mkv": return "video/x-matroska"
        case "avi": return "video/x-msvideo"
        case "md", "markdown": return "text/markdown"
        case "glb": return "model/gltf-binary"
        default: return nil
        }
    }
    /// "PNG or JPEG image".
    var title: String {
        let names = mimes.map { m -> String in
            if m.hasSuffix("/*") { return String(m.dropLast(2)) }
            return (m.split(separator: "/").last.map(String.init) ?? m).uppercased()
        }
        guard !names.isEmpty else { return "Any file" }
        return names.count == 1 ? names[0] : names.dropLast().joined(separator: ", ") + " or " + names.last!
    }
    var images: Bool { !mimes.isEmpty && mimes.allSatisfy { $0.hasPrefix("image/") } }
    var audio: Bool { !mimes.isEmpty && mimes.allSatisfy { $0.hasPrefix("audio/") } }
    var video: Bool { !mimes.isEmpty && mimes.allSatisfy { $0.hasPrefix("video/") } }
    /// A document well (PDF, maybe with images): neither only images, audio nor video.
    var document: Bool { mimes.contains("application/pdf") }
}

struct FormField: Equatable, Identifiable {
    enum Control: Equatable {
        case text(multiline: Bool)
        /// A whole number; `stepper` when both bounds are close enough to step through.
        case integer(min: Int?, max: Int?, stepper: Bool)
        case number(min: Double?, max: Double?)
        case toggle
        case choice([JSONValue])
        /// `multiple`: an array of files (up to `maxItems`); `wrapped`: each file is sent as `{"path": …}`.
        case file(FileAccept, multiple: Bool, maxItems: Int, wrapped: Bool)
        case json
    }
    var key: String
    var title: String
    var help: String?
    var control: Control
    var required: Bool
    var defaultValue: JSONValue?
    var id: String { key }
}

/// What a field holds while it is edited.
enum FieldValue: Equatable {
    case text(String)
    case toggle(Bool)
    case choice(Int?)
    case files([URL])

    var text: String { if case .text(let s) = self { return s }; return "" }
    var files: [URL] { if case .files(let f) = self { return f }; return [] }
}

struct OperationForm: Equatable {
    var fields: [FormField]
    /// The schema is not an object of properties (a union at the top, say): the whole input is one JSON field.
    var wholeJSON: Bool
    var constants: JSONObject = [:]

    /// Keys the service manages or the transport decides; never shown.
    static let hidden: Set<String> = ["out_dir", "stream"]

    init(schema: JSONObject) {
        let root = SchemaRefs(root: schema)
        let object = root.resolve(schema)
        guard object["type"]?.string == "object" || object["properties"] != nil, let properties = object["properties"]?.object,
            object["anyOf"] == nil, object["oneOf"] == nil, object["allOf"] == nil
        else {
            fields = [FormField(key: "", title: "Input", help: nil, control: .json, required: true, defaultValue: object["default"])]
            wholeJSON = true
            return
        }
        let required = object["required"]?.array?.compactMap(\.string) ?? []
        var fields: [FormField] = []
        for (key, raw) in properties where !Self.hidden.contains(key) {
            guard let property = raw.object.map(root.resolve) else { continue }
            if let value = property["const"] { constants[key] = value; continue }
            fields.append(
                FormField(
                    key: key, title: property["title"]?.string ?? Self.title(key), help: property["description"]?.string,
                    control: Self.control(property, root: root), required: required.contains(key), defaultValue: property["default"]))
        }
        // JSON objects carry no order: the content (files, then long text) first, then required fields as the schema lists
        // them, then the rest by title, switches last.
        func rank(_ f: FormField) -> (Int, Int, String) {
            let group: Int
            switch f.control {
            case .file: group = 0
            case .text(true): group = 1
            case .toggle: group = 4
            case .json: group = 5
            default: group = f.required ? 2 : 3
            }
            return (group, required.firstIndex(of: f.key) ?? Int.max, f.title.lowercased())
        }
        self.fields = fields.sorted { rank($0) < rank($1) }
        wholeJSON = false
    }

    static func control(_ p: JSONObject, root: SchemaRefs) -> FormField.Control {
        if let accept = fileAccept(p) { return .file(accept, multiple: false, maxItems: 1, wrapped: p["type"]?.string != "string") }
        if p["type"]?.string == "array", let items = p["items"]?.object.map(root.resolve), let accept = fileAccept(items) {
            let most = p["maxItems"]?.number.flatMap { $0.isFinite ? Int(max(1, min(64, $0))) : nil } ?? 16
            return .file(accept, multiple: true, maxItems: most, wrapped: items["type"]?.string != "string")
        }
        if let choices = p["enum"]?.array, !choices.isEmpty, choices.count <= 64 { return .choice(choices) }
        if p["anyOf"] != nil || p["oneOf"] != nil || p["allOf"] != nil { return .json }
        let types = p["type"]?.array?.compactMap(\.string).filter { $0 != "null" } ?? [p["type"]?.string].compactMap { $0 }
        guard types.count == 1 else { return .json }
        switch types[0] {
        case "string":
            let short = p["maxLength"]?.number.map { $0 <= 200 } ?? false
            return .text(multiline: !short && p["format"] == nil && p["pattern"] == nil)
        case "integer":
            let lo = bound(p, "minimum", "exclusiveMinimum", +1), hi = bound(p, "maximum", "exclusiveMaximum", -1)
            let stepper = lo.flatMap { l in hi.map { h in l <= h && h - l <= 1000 } } ?? false
            return .integer(min: lo, max: hi, stepper: stepper)
        case "number":
            return .number(min: p["minimum"]?.number, max: p["maximum"]?.number)
        case "boolean": return .toggle
        default: return .json
        }
    }

    private static func bound(_ p: JSONObject, _ inclusive: String, _ exclusive: String, _ nudge: Int) -> Int? {
        if let v = p[inclusive]?.number, v.isFinite, abs(v) < 1e15 { return nudge > 0 ? Int(v.rounded(.up)) : Int(v.rounded(.down)) }
        if let v = p[exclusive]?.number, v.isFinite, abs(v) < 1e15 { return (nudge > 0 ? Int(v.rounded(.down)) : Int(v.rounded(.up))) + nudge }
        return nil
    }

    static func fileAccept(_ p: JSONObject) -> FileAccept? {
        guard let note = p["x-arsenal-file"] else { return nil }
        return FileAccept(mimes: note["accept"]?.array?.compactMap(\.string) ?? [])
    }

    /// "num_layers" → "Num layers".
    static func title(_ key: String) -> String {
        let words = key.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }

    // MARK: Values

    /// The value a field starts with: the model's own setting of that name, else the schema's default.
    func initialValue(_ f: FormField, setting: JSONValue?) -> FieldValue {
        let seed = setting ?? f.defaultValue
        switch f.control {
        case .text: return .text(seed?.string ?? "")
        case .integer, .number: return .text(seed?.number.map(Self.number) ?? "")
        case .toggle: return .toggle(seed?.bool ?? false)
        // A saved setting the choices no longer list (a value once typed freely) starts at the declared default.
        case .choice(let choices):
            return .choice(seed.flatMap { s in choices.firstIndex(of: s) } ?? f.defaultValue.flatMap { choices.firstIndex(of: $0) } ?? (f.required ? 0 : nil))
        case .file: return .files([])
        case .json: return .text(seed.flatMap(Self.pretty) ?? "")
        }
    }

    static func number(_ v: Double) -> String {
        v.rounded() == v && abs(v) < 1e15 ? String(Int(v)) : String(v)
    }
    static func pretty(_ v: JSONValue) -> String? {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return (try? encoder.encode(v)).flatMap { String(data: $0, encoding: .utf8) }
    }

    /// A field's problem before anything is sent: empty and required, or not a number / not JSON. The schema's own rules
    /// are the service validator's to judge.
    struct Problem: Error, Equatable { var key: String; var message: String }

    /// The request body: every filled field, files as the staged paths `stage` returns (in the order dropped).
    func input(_ values: [String: FieldValue], stage: (URL) throws -> String) throws -> JSONObject {
        if wholeJSON {
            let text = values[""]?.text ?? ""
            guard let data = text.data(using: .utf8), case .object(let object)? = try? JSONDecoder().decode(JSONValue.self, from: data) else {
                throw Problem(key: "", message: "The input must be a JSON object.")
            }
            return object
        }
        var input = constants
        for f in fields {
            let value = values[f.key] ?? initialValue(f, setting: nil)
            let encoded: JSONValue?
            switch (f.control, value) {
            case (.text, .text(let s)):
                encoded = s.isEmpty && !f.required ? nil : .string(s)
            case (.integer, .text(let s)):
                let t = s.trimmingCharacters(in: .whitespaces)
                if t.isEmpty {
                    encoded = nil
                } else {
                    guard let n = Int(t) else { throw Problem(key: f.key, message: "\(f.title) must be a whole number.") }
                    encoded = .number(Double(n))
                }
            case (.number, .text(let s)):
                let t = s.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")
                if t.isEmpty {
                    encoded = nil
                } else {
                    guard let n = Double(t), n.isFinite else { throw Problem(key: f.key, message: "\(f.title) must be a number.") }
                    encoded = .number(n)
                }
            case (.toggle, .toggle(let on)):
                encoded = .bool(on)
            case (.choice(let choices), .choice(let i)):
                encoded = i.flatMap { choices.indices.contains($0) ? choices[$0] : nil }
            case (.file(_, let multiple, _, let wrapped), .files(let urls)):
                let staged = try urls.map { url -> JSONValue in
                    let path = try stage(url)
                    return wrapped ? .object(["path": .string(path)]) : .string(path)
                }
                encoded = staged.isEmpty ? nil : multiple ? .array(staged) : staged[0]
            case (.json, .text(let s)):
                if s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    encoded = nil
                } else {
                    guard let data = s.data(using: .utf8), let v = try? JSONDecoder().decode(JSONValue.self, from: data) else {
                        throw Problem(key: f.key, message: "\(f.title) is not valid JSON.")
                    }
                    encoded = v
                }
            default: encoded = nil
            }
            if let encoded {
                input[f.key] = encoded
            } else if f.required {
                if case .file = f.control { throw Problem(key: f.key, message: "Add \(f.title.lowercased()) first.") }
                throw Problem(key: f.key, message: "\(f.title) is needed.")
            }
        }
        return input
    }
}

/// Local `$ref`s ("#/$defs/…") resolved against the operation's own schema, as the service's validator resolves them.
struct SchemaRefs {
    let root: JSONObject
    func resolve(_ schema: JSONObject) -> JSONObject {
        var current = schema
        var hops = 0
        while let ref = current["$ref"]?.string, hops < 16 {
            hops += 1
            guard ref == "#" || ref.hasPrefix("#/") else { return current }
            var target = JSONValue.object(root)
            for part in ref.dropFirst(2).split(separator: "/") {
                target = target[String(part).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")] ?? .null
            }
            guard var resolved = target.object else { return current }
            // Sibling keywords (title, description, default) beside a $ref describe this use of it.
            for (k, v) in current where k != "$ref" { resolved[k] = v }
            current = resolved
        }
        return current
    }
}
