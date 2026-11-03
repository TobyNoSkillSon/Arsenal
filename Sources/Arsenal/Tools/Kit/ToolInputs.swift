// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// The tool kit's semantic inputs, drawn wherever an operation's schema names them (`x-arsenal-control`): a language
// pop-up, chips for short lists (labels, stems), and the fields an answer must have (a JSON Schema built as named
// fields, its JSON one click away).

/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
private typealias InputState<T> = SwiftUI.State<T>

/// Languages offered when an operation takes any language code: the most spoken first, then alphabetical by name.
enum LanguageList {
    static let common = [
        "en", "zh", "es", "fr", "de", "ja", "ko", "pt", "it", "ru", "ar", "hi", "nl", "pl", "tr", "sv", "uk", "cs", "vi", "id", "th", "he", "el", "da",
        "fi", "no", "ro", "hu", "bg", "hr", "sk", "sl", "lt", "lv", "et", "ms", "fa", "bn", "ta", "ur"
    ]
    /// A language a model reports detecting, or nil when it detected none ("und", "unknown", empty), so a result never
    /// reads "Unknown language".
    static func detected(_ code: String?) -> String? {
        guard let code = code?.trimmingCharacters(in: .whitespacesAndNewlines), !code.isEmpty,
            !["und", "unknown", "auto", "none", "null", "zxx", "mis", "mul", "n/a"].contains(code.lowercased())
        else { return nil }
        return code
    }
    /// "en" → English, "english" → English, "EN-US" → American English.
    static func name(_ code: String) -> String {
        let named = VoiceChoice.languageName(code)
        if !named.isEmpty, named.lowercased() != code.lowercased() { return named }
        return FieldControl.choiceTitle(code)
    }
}

/// A language pop-up: Automatic (nothing sent) unless the operation requires one, then the schema's own list, else
/// common languages; a code the window was given that is not listed stays listed by name.
struct LanguagePicker: View {
    let field: ToolField
    let choices: [String]?
    @ObservedObject var session: ToolSession

    var body: some View {
        let current = session.value(field.key).text
        let list = choices ?? LanguageList.common
        Picker(field.title, selection: Binding(get: { current }, set: { session.set(field.key, .text($0)) })) {
            if !field.required { Text(field.key == "target_lang" ? "Choose language" : "Automatic").tag("") }
            if field.required, current.isEmpty { Text("Choose language").tag("").selectionDisabled() }
            if !current.isEmpty, !list.contains(current) { Text(LanguageList.name(current)).tag(current) }
            ForEach(list, id: \.self) { code in Text(LanguageList.name(code)).tag(code) }
        }
        .labelsHidden().pickerStyle(.menu).frame(maxWidth: .infinity, alignment: .leading)
        .controlID("field-\(field.key)")
    }
}

/// Short strings as chips; Return or a comma adds what is typed, × removes one. The value is the comma-joined list.
struct TagField: View {
    let field: ToolField
    @ObservedObject var session: ToolSession
    var placeholder = "Type and press Return"
    @InputState private var draft = ""
    @Environment(\.palette) private var p
    @Environment(\.isEnabled) private var enabled

    private var tags: [String] { ToolForm.items(session.value(field.key).text, separators: ",\n") }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !tags.isEmpty {
                FlowLayout(spacing: 5) {
                    ForEach(Array(tags.enumerated()), id: \.offset) { i, tag in chip(tag, at: i) }
                }
            }
            TextField(field.title, text: $draft, prompt: Text(tags.isEmpty ? placeholder : "Add another").foregroundStyle(p.t3.swiftUI))
                .textFieldStyle(.plain).font(.system(size: 13))
                .padding(.horizontal, 8).frame(height: 26)
                .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
                .onSubmit(add)
                .onChange(of: draft) { _, text in if text.contains(",") { add() } }
                .controlID("field-\(field.key)")
        }
    }

    /// A chip's words: stems capitalised as their result rows name them ("Vocals"); the value sent keeps the typed text.
    static func shown(_ tag: String, key: String) -> String {
        key == "stems" ? tag.prefix(1).uppercased() + tag.dropFirst() : tag
    }

    private func chip(_ tag: String, at index: Int) -> some View {
        HStack(spacing: 3) {
            Text(Self.shown(tag, key: field.key)).font(.system(size: 12, weight: .medium)).lineLimit(1)
            if enabled {
                Button {
                    var list = tags
                    list.remove(at: index)
                    session.set(field.key, .text(list.joined(separator: ", ")))
                } label: {
                    Image(systemName: "xmark").font(.system(size: 8, weight: .bold)).frame(width: 14, height: 14).contentShape(Rectangle())
                }.buttonStyle(.plain).windowHelp("Remove \(tag)").accessibilityLabel("Remove \(tag)").controlID("remove-tag-\(tag)")
            }
        }
        .foregroundStyle(p.t1.swiftUI).padding(.leading, 8).padding(.trailing, enabled ? 3 : 8).frame(height: 22)
        .background(p.selFill.swiftUI, in: RoundedRectangle(cornerRadius: 6))
    }

    private func add() {
        let new = ToolForm.items(draft, separators: ",\n")
        draft = ""
        guard !new.isEmpty else { return }
        var list = tags
        for tag in new where !list.contains(tag) { list.append(tag) }
        session.set(field.key, .text(list.joined(separator: ", ")))
    }
}

/// Strings one per row (passages to rank, texts to embed): each its own field with ×, and a field that adds one (a
/// pasted block of lines adds a row per line). The value is the rows joined by newlines.
struct LinesEditor: View {
    let field: ToolField
    @ObservedObject var session: ToolSession
    @InputState private var draft = ""
    @Environment(\.palette) private var p
    @Environment(\.isEnabled) private var enabled

    private var rows: [String] { ToolForm.items(session.value(field.key).text, separators: "\n") }
    private func write(_ rows: [String]) { session.set(field.key, .text(rows.joined(separator: "\n"))) }

    var body: some View {
        let rows = self.rows
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(rows.enumerated()), id: \.offset) { i, row in
                HStack(alignment: .top, spacing: 4) {
                    TextField(
                        field.title,
                        text: Binding(
                            get: { row },
                            set: { new in
                                var list = self.rows
                                guard list.indices.contains(i) else { return }
                                list[i] = new.replacingOccurrences(of: "\n", with: " ")
                                write(list)
                            }), axis: .vertical
                    )
                    .textFieldStyle(.plain).font(.system(size: 13)).lineLimit(1...4)
                    .controlID("row-\(field.key)-\(i)")
                    if enabled {
                        Button {
                            var list = self.rows
                            if list.indices.contains(i) { list.remove(at: i) }
                            write(list)
                        } label: {
                            Image(systemName: "xmark").font(.system(size: 9, weight: .bold)).frame(width: 18, height: 18).contentShape(Rectangle())
                        }.buttonStyle(.plain).foregroundStyle(p.t3.swiftUI).windowHelp("Remove").accessibilityLabel("Remove \(row)")
                    }
                }
                .padding(.horizontal, 8).padding(.vertical, 6)
                .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
            }
            TextField(field.title, text: $draft, prompt: Text(field.key == "documents" ? "Add a passage" : "Add a line").foregroundStyle(p.t3.swiftUI), axis: .vertical)
                .textFieldStyle(.plain).font(.system(size: 13)).lineLimit(1...4)
                .padding(.horizontal, 8).padding(.vertical, 6)
                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(p.raisedBorder.swiftUI, style: StrokeStyle(lineWidth: 1, dash: [4, 3])))
                .onSubmit(add)
                .onChange(of: draft) { _, text in if text.contains("\n") { add() } }
                .controlID("field-\(field.key)")
        }
    }

    private func add() {
        let new = ToolForm.items(draft, separators: "\n")
        draft = ""
        guard !new.isEmpty else { return }
        write(rows + new)
    }
}

/// Lays chips out in rows, wrapping at the container's width.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, row: CGFloat = 0, widest: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width { y += row + spacing; x = 0; row = 0 }
            x += size.width + spacing; row = max(row, size.height); widest = max(widest, x - spacing)
        }
        return CGSize(width: proposal.width ?? widest, height: y + row)
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, row: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX { y += row + spacing; x = bounds.minX; row = 0 }
            view.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing; row = max(row, size.height)
        }
    }
}

// MARK: Schema

/// One field of the answer a schema asks for.
struct SchemaField: Equatable, Identifiable {
    enum Kind: String, CaseIterable, Identifiable {
        case text, number, integer, boolean, list
        var id: String { rawValue }
        var title: String {
            switch self {
            case .text: return "Text"
            case .number: return "Number"
            case .integer: return "Whole number"
            case .boolean: return "Yes or no"
            case .list: return "List of text"
            }
        }
        var schema: JSONValue {
            switch self {
            case .text: return ["type": "string"]
            case .number: return ["type": "number"]
            case .integer: return ["type": "integer"]
            case .boolean: return ["type": "boolean"]
            case .list: return ["type": "array", "items": ["type": "string"]]
            }
        }
        init?(_ schema: JSONObject) {
            switch schema["type"]?.string {
            case "string": self = .text
            case "number": self = .number
            case "integer": self = .integer
            case "boolean": self = .boolean
            case "array" where schema["items"]?["type"]?.string == "string": self = .list
            default: return nil
            }
        }
    }
    var id = UUID()
    var name: String
    var kind: Kind

    static func == (a: SchemaField, b: SchemaField) -> Bool { a.name == b.name && a.kind == b.kind }

    /// The fields of a flat object schema (every property a plain type); nil for anything richer, which is edited as JSON.
    static func fields(_ text: String) -> [SchemaField]? {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty { return [] }
        guard let data = t.data(using: .utf8), case .object(let schema)? = try? JSONDecoder().decode(JSONValue.self, from: data),
            schema["type"]?.string == "object", let properties = schema["properties"]?.object
        else { return nil }
        let order = schema["required"]?.array?.compactMap(\.string) ?? []
        var fields: [SchemaField] = []
        for name in order + properties.keys.sorted() where !fields.contains(where: { $0.name == name }) {
            guard let p = properties[name]?.object, let kind = Kind(p) else { return nil }
            fields.append(SchemaField(name: name, kind: kind))
        }
        return fields
    }

    /// Every named field, required, nothing else allowed.
    static func schema(_ fields: [SchemaField]) -> String {
        let named = fields.filter { !$0.name.trimmingCharacters(in: .whitespaces).isEmpty }
        guard !named.isEmpty else { return "" }
        var properties: JSONObject = [:]
        for f in named { properties[f.name.trimmingCharacters(in: .whitespaces)] = f.kind.schema }
        let schema: JSONValue = [
            "type": "object", "properties": .object(properties),
            "required": .array(named.map { .string($0.name.trimmingCharacters(in: .whitespaces)) }), "additionalProperties": false
        ]
        return OperationForm.pretty(schema) ?? ""
    }
}

/// The fields the answer must have: a name and a type per row, + to add one; the JSON Schema they make is one click
/// away (and a schema richer than flat fields opens as JSON).
struct SchemaEditor: View {
    let key: String
    @Binding var text: String
    /// No starter row: an optional extra shows only its Add until a field is added.
    var startsEmpty = false
    @InputState private var rows: [SchemaField] = []
    @InputState private var raw = false
    @InputState private var loaded: String?
    @Environment(\.palette) private var p
    @Environment(\.isEnabled) private var enabled

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if raw || SchemaField.fields(text) == nil {
                JSONTextEditor(text: $text, height: 120, id: "field-\(key)")
            } else {
                ForEach($rows) { $row in
                    HStack(spacing: 6) {
                        TextField("Name", text: $row.name, prompt: Text("name").foregroundStyle(p.t3.swiftUI)).textFieldStyle(.plain).font(.system(size: 12.5))
                            .padding(.horizontal, 7).frame(height: 24)
                            .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 6))
                            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
                            .controlID("schema-name-\(rows.firstIndex(where: { $0.id == row.id }) ?? 0)")
                        Picker("Type", selection: $row.kind) { ForEach(SchemaField.Kind.allCases) { Text($0.title).tag($0) } }
                            .labelsHidden().pickerStyle(.menu).fixedSize()
                        Button {
                            rows.removeAll { $0.id == row.id }
                        } label: {
                            Image(systemName: "minus.circle").font(.system(size: 12)).frame(width: 20, height: 22).contentShape(Rectangle())
                        }.buttonStyle(.plain).foregroundStyle(p.t3.swiftUI).windowHelp("Remove this field").accessibilityLabel("Remove \(row.name)")
                    }
                }
                HStack(spacing: 8) {
                    Button {
                        rows.append(SchemaField(name: "", kind: .text))
                    } label: {
                        Label(rows.isEmpty ? "Add a field" : "Add another", systemImage: "plus").font(.system(size: 12, weight: .medium))
                            .foregroundStyle((enabled ? p.accentText : p.t4).swiftUI).frame(height: 22).contentShape(Rectangle())
                    }.buttonStyle(.plain).controlID("schema-add")
                    Spacer(minLength: 0)
                }
            }
        }
        .overlay(alignment: .bottomTrailing) {
            // The schema's JSON, one click away (a schema richer than flat fields opens as JSON by itself).
            if SchemaField.fields(text) != nil {
                Button {
                    raw.toggle()
                    if !raw { rows = SchemaField.fields(text) ?? rows }
                } label: {
                    Image(systemName: "chevron.left.forwardslash.chevron.right").font(.system(size: 10)).frame(width: 22, height: 22).contentShape(Rectangle())
                }.buttonStyle(.plain).foregroundStyle((raw ? p.accentText : p.t3).swiftUI).windowHelp(raw ? "Edit as fields" : "Edit as JSON")
                    .offset(y: raw ? 24 : 0).controlID("schema-json")
            }
        }
        .padding(.bottom, raw ? 24 : 0)
        .onAppear { load(text) }
        .onChange(of: key) { _, _ in load(text) }
        .onChange(of: rows) { _, rows in
            let schema = SchemaField.schema(rows)
            loaded = schema
            if schema != text { text = schema }
        }
        .onChange(of: text) { _, text in if text != loaded, !raw { load(text) } }
    }

    private func load(_ text: String) {
        loaded = text
        rows = SchemaField.fields(text) ?? []
        if rows.isEmpty, text.isEmpty, !startsEmpty { rows = [SchemaField(name: "", kind: .text)] }
    }
}

/// JSON typed in monospace, on the field ground.
struct JSONTextEditor: View {
    @Binding var text: String
    var height: CGFloat = 96
    let id: String
    @Environment(\.palette) private var p
    var body: some View {
        TextEditor(text: $text)
            .font(.system(size: 11.5, design: .monospaced)).scrollContentBackground(.hidden)
            .padding(.horizontal, 6).padding(.vertical, 6)
            .frame(height: height)
            .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
            .controlID(id)
    }
}
