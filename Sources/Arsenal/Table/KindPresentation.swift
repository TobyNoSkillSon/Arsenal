// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// How a kind is shown: the group a model sits in (table, Get more models, its sidebar) and its icon. Models of the chat
/// kind that do not chat (no chat model ships:) are shown by what they do, from their folder's `tasks`:
/// reading documents, translating, extracting. The service's kind stays the model's kind (runners, figures, the saved
/// group order); only the presentation splits.
enum KindPresentation {
    /// A shown group made from a service kind: its id, title, icon, and the tasks that place a model in it.
    struct Split: Equatable {
        var id: String
        var parent: String
        var title: String
        var icon: String
        var tasks: [String]
    }
    /// In this order after their parent kind.
    static let splits: [Split] = [
        Split(id: "documents", parent: "chat", title: "Documents", icon: "doc.text.viewfinder", tasks: ["read_document"]),
        Split(id: "translation", parent: "chat", title: "Translation", icon: "translate", tasks: ["translate"]),
        Split(id: "extraction", parent: "chat", title: "Extraction", icon: "curlybraces", tasks: ["extract"]),
        Split(id: "separation", parent: "audio", title: "Separation", icon: "slider.vertical.3", tasks: ["separate_audio"])
    ]
    /// What a kind's own models do: a model doing none of it shows in a split.
    static let own: [String: Set<String>] = ["chat": ["chat"], "audio": ["transcribe", "diarize"]]
    /// Tasks another kind's group names: a model doing none of its own kind's things whose task is here shows in that
    /// kind's group (Smart Turn detects the end of a spoken turn: it belongs with Speech, not Transcription).
    static let elsewhere: [String: String] = ["end_of_turn": "voices"]
    /// The groups named by what their models do, as the tool they open names it, short enough for the sidebar; the
    /// service's ids and titles stay (the API, MCP and the saved group order use them).
    static let names: [String: (title: String, icon: String?)] = [
        "voices": ("Speech", nil), "sound": ("Music & Sounds", "music.note"), "enhancement": ("Audio Cleanup", nil),
        "audio": ("Transcription", "quote.bubble"), "perception": ("Objects & Depth", nil), "restoration": ("Upscale & Smooth", nil),
        "embeddings": ("Search", "text.magnifyingglass")
    ]
    /// A service kind under its shown name and icon.
    static func named(_ k: KindSpec) -> KindSpec {
        guard let name = names[k.id] else { return k }
        var s = k
        s.title = name.title; s.titleSingular = name.title
        if let icon = name.icon { s.icon = icon }
        return s
    }
    /// The shown name of a kind id ("audio" → "Transcription"), nil for one without its own.
    static func title(_ id: String) -> String? { names[id]?.title ?? splits.first { $0.id == id }?.title }

    /// The shown kind of a model: its kind, or the split its first declared task places it in when it does none of its
    /// kind's own things (a chat-kind model with no `chat` task, an audio model that only separates).
    static func id(kind: String, tasks: [String]) -> String {
        guard let mine = own[kind], !tasks.contains(where: mine.contains) else { return kind }
        for task in tasks {
            if let split = splits.first(where: { $0.parent == kind && $0.tasks.contains(task) }) { return split.id }
            if let other = elsewhere[task], other != kind { return other }
        }
        return kind
    }
    /// The shown kind, with a folder's own kind joined to the shown group that has its title (`joined`).
    static func id(kind: String, tasks: [String], kinds: [KindSpec]) -> String {
        let own = id(kind: kind, tasks: tasks)
        return own == kind ? joined(kind, kinds: kinds) ?? kind : own
    }
    /// A folder's own kind (one Arsenal has no stat kind for) titled as a group Arsenal's kinds already show joins that
    /// group, so no two groups share a title: the kit kind local.document ("Documents": GLM-OCR, Manga OCR) shows in
    /// Documents beside TeleOCR, under its description.
    static func joined(_ kind: String, kinds: [KindSpec]) -> String? {
        // A folder kind that declares its own figure columns keeps its own group: their units are its own.
        guard ModelStatKind(rawValue: kind) == nil, let spec = kinds.first(where: { $0.id == kind }), spec.columns.isEmpty else { return nil }
        let title = spec.title
        return expand(kinds.filter { ModelStatKind(rawValue: $0.id) != nil })
            .first { $0.id != kind && $0.title.caseInsensitiveCompare(title) == .orderedSame }?.id
    }
    /// The service kind a shown kind belongs to (itself for a service kind).
    static func parent(_ id: String) -> String { splits.first { $0.id == id }?.parent ?? id }
    /// The service kind whose groups show a model: its shown kind's parent (`shown`: its shown kind, nil for its own).
    static func placed(kind: String, shown: String?) -> String { parent(shown ?? kind) }
    static func isSplit(_ id: String) -> Bool { splits.contains { $0.id == id } }

    /// The service's kinds with each split after its parent, as specs of their own (the parent's columns and order).
    static func expand(_ kinds: [KindSpec]) -> [KindSpec] {
        kinds.flatMap { k in [named(k)] + splits.filter { $0.parent == k.id }.map { spec($0, parent: k) } }
    }
    static func spec(_ split: Split, parent k: KindSpec) -> KindSpec {
        var s = k
        s.id = split.id; s.title = split.title; s.titleSingular = split.title; s.icon = split.icon
        return s
    }

    /// A service kind's rows as the groups they show in: the kind's own first, then each split that has rows.
    /// `shown`: a row's shown kind by id (absent: its own kind).
    static func groups(_ k: KindSpec, rows: [ModelRow], shown: [String: String]) -> [(kind: KindSpec, rows: [ModelRow])] {
        let own = rows.filter { (shown[$0.id] ?? $0.kind) == k.id }
        var out: [(kind: KindSpec, rows: [ModelRow])] = own.isEmpty ? [] : [(named(k), own)]
        for split in splits where split.parent == k.id {
            let mine = rows.filter { shown[$0.id] == split.id }
            if !mine.isEmpty { out.append((spec(split, parent: k), mine)) }
        }
        return out
    }
}

/// A parameter count as both windows print it, whatever the folder wrote ("2038M", "3.70B", "8.1M"). The size in the
/// model's name when it carries one ("Qwen3 ASR 1.7B" shows 1.7B, as Vella and the model's makers say it, not the
/// counted 2.04B); otherwise one decimal in B from a billion ("3.1B", trailing zero dropped: "4B") and whole numbers in M
/// below ("782M", "8M").
enum ParamsText {
    static func value(_ text: String) -> Double? {
        let t = text.uppercased().trimmingCharacters(in: .whitespaces)
        let scale: Double = t.hasSuffix("B") ? 1e9 : t.hasSuffix("M") ? 1e6 : t.hasSuffix("K") ? 1e3 : 1
        guard let n = Double(t.trimmingCharacters(in: .letters)), n.isFinite, n > 0 else { return nil }
        return n * scale
    }
    /// The size a name carries as its own word ("1.7B" in "Qwen3 ASR 1.7B", "82M" in "Kokoro 82M"), when it is within
    /// a factor of two of the counted parameters (a version number or a stray word never stands in for the size).
    static func named(_ name: String, near counted: Double?) -> String? {
        let words = name.split(whereSeparator: { $0 == " " || $0 == "\u{00b7}" }).map(String.init)
        for word in words.reversed() {
            guard word.range(of: #"^[0-9]+(\.[0-9]+)?[BMK]$"#, options: .regularExpression) != nil, let v = value(word) else { continue }
            if let counted, v / counted > 2 || counted / v > 2 { continue }
            return word
        }
        return nil
    }
    static func display(_ text: String, name: String? = nil) -> String {
        let counted = value(text)
        if let name, let shown = named(name, near: counted) { return shown }
        guard let v = counted else { return text }
        if v >= 999.5e6 {
            // One decimal in B; 999.6M rounds to 1B.
            var s = String(format: "%.1f", (v / 1e8).rounded() / 10)
            if s.hasSuffix(".0") { s.removeLast(2) }
            return s + "B"
        }
        if v >= 999.5e3 { return String(Int((v / 1e6).rounded())) + "M" }
        return String(max(1, Int((v / 1e3).rounded()))) + "K"
    }
}
