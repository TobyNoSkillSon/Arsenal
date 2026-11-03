// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// The tool kit's options: one form for every tool window (Make an Image, Make a Sound, Remove Background, a model's own
// operations), read from the chosen operation's input schema. Properties the kit knows by name get their own control:
// the main text (`prompt` or `text`), the size (`width` and `height`, or a `resolution` list), the voice, the seed, a
// length in seconds, reference images, a voice to clone (`reference`: a clip and what it says) and free-text notes
// (style, voice description, what to avoid). Every other property gets the generic control for its schema
// (OperationForm), so an input a model folder adds or corrects appears without a change here. The service's own
// validator judges the values.

struct ToolField: Equatable, Identifiable {
    enum Control: Equatable {
        /// The main text: what to make or what to say.
        case main(placeholder: String)
        /// A voice id, chosen from the model's voices when it lists them.
        case voice
        case seed
        /// A length in seconds, within the schema's bounds.
        case seconds(min: Double, max: Double)
        /// A short free text under its title (style, a voice description, what to avoid).
        case note
        /// An audio clip to speak like, with what the clip says (`{"path", "text"}`); `transcript` nil: the object takes
        /// no text; true: it requires one.
        case reference(transcript: Bool?)
        /// The picture's size as one pop-up: `width` and `height` within their bounds (`free`), else the `resolution`
        /// list. Its value is "1024x1024".
        case size(ImageSizes)
        /// Images sent inline (base64 strings), up to `max`.
        case images(max: Int)
        /// A language code (`x-arsenal-control: language`): the schema's own list when it has one, else common languages;
        /// Automatic (nothing sent) unless required.
        case language(choices: [String]?)
        /// A song's words (`x-arsenal-control: lyrics`), off while the instrumental switch is on.
        case lyrics
        /// Short strings as chips (labels, stems): an array of strings.
        case tags
        /// Strings one per line (documents to rank, texts to embed): an array of strings.
        case lines
        /// Page numbers as typed ("1-3, 5"), sent as zero-based indices.
        case pages
        /// Numbers as typed ("0.25, 0.5"): an array of numbers within the schema's bounds.
        case numbers(min: Double?, max: Double?)
        /// A JSON Schema the answer must follow (`x-arsenal-control: json-schema`), edited as named fields.
        case schema
        /// Points or boxes over the image or video field `image` (`image-points`, `image-boxes`), normalized to its
        /// original extent. A capability window draws them over its subject; elsewhere they are edited as JSON.
        case points(image: String)
        case boxes(image: String)
        /// Anything else, as the generic operation form shows it.
        case generic(FormField.Control)
    }
    var key: String
    var title: String
    var help: String?
    var control: Control
    var required: Bool
    var defaultValue: JSONValue?
    /// The main text is sent as a one-item list (Translate's `text`).
    var sendsList = false
    /// A window's words for a choice's values, by the value as written ("translate" → "English translation", 1024 →
    /// "1024 px"); a value without one reads as the kit names it.
    var choiceTitles: [String: String] = [:]
    /// The unit after a number's figure ("4×").
    var unit: String?
    /// A field list that starts with no row, only its Add (an optional extra, not the window's main ask).
    var startsEmpty = false
    var id: String { key }
    /// A row of the compact grid (label left, control right) rather than a block with its title above.
    var compact: Bool {
        switch control {
        case .voice, .seed, .seconds, .size, .language, .pages: return true
        case .generic(let c):
            switch c {
            case .integer, .number, .choice, .text(false): return true
            default: return false
            }
        default: return false
        }
    }
    var isMain: Bool {
        if case .main = control { return true }; return false
    }
    /// Drawn over the window's subject rather than among the options.
    var isSpatial: Bool {
        switch control {
        case .points, .boxes: return true
        default: return false
        }
    }
}

/// The sizes a size field offers and how it is sent.
struct ImageSizes: Equatable {
    var sizes: [ImageSize]
    /// Sent as `width` and `height` (any multiple of 16 within the bounds); else as `resolution` from the list.
    var free: Bool
    var widthRange: ClosedRange<Int>?
    var heightRange: ClosedRange<Int>?
    /// The operation edits reference images and its size is optional: the pop-up starts at "Same as reference" (an
    /// empty value, so no size is sent and the model keeps the reference's size).
    var reference = false
    /// A size outside the list the operation still takes (a model setting's free size).
    func accepts(_ s: ImageSize) -> Bool {
        if !free { return sizes.contains(s) }
        guard let w = widthRange, let h = heightRange else { return false }
        return w.contains(s.width) && h.contains(s.height) && s.width % 16 == 0 && s.height % 16 == 0
    }
    /// The size a window starts with: none for a reference edit, else 1024 × 1024 when offered, else the largest listed.
    var preferred: ImageSize? { reference ? nil : sizes.contains(.square1024) ? .square1024 : sizes.last }
    static let sameAsReference = "Same as reference"
}

struct ImageSize: Hashable, Identifiable {
    var width: Int
    var height: Int
    var id: String { resolution }
    var resolution: String { "\(width)x\(height)" }
    var title: String { "\(width) \u{00d7} \(height)" }
    init(width: Int, height: Int) { self.width = width; self.height = height }
    init?(resolution: String) {
        let parts = resolution.split(separator: "x").compactMap { Int($0) }
        guard parts.count == 2 else { return nil }
        self.init(width: parts[0], height: parts[1])
    }
    static let square1024 = ImageSize(width: 1024, height: 1024)
    /// Free sizes offered when the operation takes width and height: squares, then 4:3 and 16:9 both ways. The images
    /// settings reference (kinds/images/settings.toml) offers the same shapes.
    static let free: [ImageSize] = [
        .init(width: 512, height: 512), .init(width: 768, height: 768), .init(width: 1024, height: 1024), .init(width: 1152, height: 864),
        .init(width: 864, height: 1152), .init(width: 1344, height: 768), .init(width: 768, height: 1344)
    ]
}

struct ToolForm: Equatable {
    var fields: [ToolField]
    /// A window's own changes to one field (its title, its choices' words, its control).
    mutating func update(_ key: String, _ change: (inout ToolField) -> Void) {
        if let i = fields.firstIndex(where: { $0.key == key }) { change(&fields[i]) }
    }
    /// A whole number with a wide range offered as the values that make sense (a 3D model's detail, a texture size):
    /// a pop-up of those inside the declared range, titled by `title`.
    mutating func offer(_ key: String, _ values: [Int], title: (Int) -> String) {
        update(key) { f in
            guard case .generic(.integer(let lo, let hi, _)) = f.control else { return }
            let inside = values.filter { (lo ?? .min) <= $0 && $0 <= (hi ?? .max) }
            guard inside.count > 1 else { return }
            f.control = .generic(.choice(inside.map { .number(Double($0)) }))
            f.choiceTitles = Dictionary(uniqueKeysWithValues: inside.map { (String($0), title($0)) })
        }
    }
    /// Properties with one allowed value: not shown, always sent as that value.
    var fixed: JSONObject = [:]
    /// The schema is not an object of properties (a union at the top): the whole input is one JSON field.
    var wholeJSON = false
    /// Why the tool cannot run with this schema (a size field with nothing to offer); nil when it can.
    var unsupported: String?

    /// Keys the service manages or the transport decides; never shown.
    static let hidden: Set<String> = OperationForm.hidden
    /// The key holding a clip's transcript in the window's values.
    static func transcriptKey(_ key: String) -> String { key + ".text" }
    /// Keys that take reference images inline.
    static let imageKeys = ["reference_images", "images", "image_references"]

    /// `output`: what the operation makes ("audio", "image", …), which names the main text's placeholder. `main`: the
    /// keys a window takes as its main text, with their placeholders (nil: `prompt` or `text`, as Make an Image and
    /// Make a Sound use them); `titles`: a window's own names for its options.
    init(schema: JSONObject, operation: String = "", output: String = "", main: [String: String]? = nil, titles: [String: String] = [:]) {
        let root = SchemaRefs(root: schema)
        let object = root.resolve(schema)
        // Only a schema with no properties at all is one JSON field; a conditional requirement beside the properties
        // (a voice or a clip to clone) is the validator's to explain, and each property keeps its own control.
        guard object["properties"]?.object != nil else {
            let generic = OperationForm(schema: schema)
            fields = generic.fields.map {
                ToolField(key: $0.key, title: $0.title, help: $0.help, control: .generic($0.control), required: true, defaultValue: $0.defaultValue)
            }
            wholeJSON = generic.wholeJSON
            return
        }
        let properties = (object["properties"]?.object ?? [:]).compactMapValues { $0.object.map(root.resolve) }
        let required = object["required"]?.array?.compactMap(\.string) ?? []
        var fields: [ToolField] = [], fixed: JSONObject = [:]
        var size = Self.size(properties)
        if let found = size, !found.covers.contains(where: { required.contains($0) }), properties["images"]?["type"]?.string == "array" {
            size?.sizes.reference = true
        }
        if let size {
            fields.append(
                ToolField(key: size.key, title: "Size", help: nil, control: .size(size.sizes), required: size.covers.contains { required.contains($0) }, defaultValue: nil))
            if size.sizes.sizes.isEmpty {
                unsupported = "This model\u{2019}s image operation declares no size Arsenal can offer (its schema\u{2019}s bounds are empty or reversed)."
            }
        }
        for (key, p) in properties where !Self.hidden.contains(key) && !(size?.covers.contains(key) ?? false) {
            // One allowed value is no choice: the window shows nothing and sends that value.
            if let only = p["const"] ?? (p["enum"]?.array?.count == 1 ? p["enum"]?.array?.first : nil) {
                fixed[key] = only; continue
            }
            var control = Self.control(key, p, root: root, operation: operation, output: output)
            var sendsList = false
            if let main {
                // The window names its main text; any other `prompt` or `text` is a note under its title.
                if let placeholder = main[key] {
                    let items = p["items"]?.object.map(root.resolve)
                    if case .main = control {
                        control = .main(placeholder: placeholder)
                    } else if p["type"]?.string == "string" {
                        control = .main(placeholder: placeholder)
                    } else if p["type"]?.string == "array", items?["type"]?.string == "string" {
                        control = .main(placeholder: placeholder); sendsList = true
                    }
                } else if case .main = control {
                    control = .note
                }
            }
            var field = ToolField(
                key: key, title: titles[key] ?? p["title"]?.string ?? Self.titles[key] ?? OperationForm.title(key), help: p["description"]?.string,
                control: control, required: required.contains(key), defaultValue: p["default"])
            field.sendsList = sendsList
            fields.append(field)
        }
        self.fields = fields.sorted { Self.rank($0) < Self.rank($1) }
        self.fixed = fixed
    }

    /// Reading order: the main text, the files the operation works on, the compact settings (size first, seed last), the
    /// clip to clone and reference images, notes, switches, then whatever a plain control cannot express.
    private static func rank(_ f: ToolField) -> (Int, Int, String) {
        let order = ["size", "voice", "duration", "steps", "guidance", "speed", "language", "n"]
        switch f.control {
        case .main: return (0, 0, f.key)
        case .lyrics: return (0, 1, f.key)
        case .generic(.file): return (0, 2, f.title.lowercased())
        case .tags, .lines: return (0, 3, f.title.lowercased())
        case .seed: return (1, 99, f.key)
        case .schema: return (3, 10, f.key)
        case .numbers: return (2, 30, f.title.lowercased())
        case .points, .boxes: return (6, 0, f.key)
        case .reference, .images: return (2, 0, f.key)
        case .note: return (3, ["instructions", "description", "negative_prompt"].firstIndex(of: f.key) ?? 9, f.title.lowercased())
        case .generic(.toggle): return (4, 0, f.title.lowercased())
        case .generic(.json): return (5, 0, f.title.lowercased())
        default:
            if f.compact { return (1, order.firstIndex(of: f.key) ?? 50, f.title.lowercased()) }
            return (2, 20, f.title.lowercased())
        }
    }

    static let titles: [String: String] = [
        "duration": "Length", "negative_prompt": "Avoid", "instructions": "Style", "description": "Voice description", "reference": "Voice to clone",
        "guidance": "Guidance", "speed": "Speed", "language": "Language", "voice": "Voice", "seed": "Seed", "steps": "Steps", "n": "Images",
        "reference_images": "Reference images", "image_references": "Reference images", "source_lang": "From", "target_lang": "To", "lyrics": "Lyrics",
        "instrumental": "Instrumental", "labels": "Labels", "stems": "Stems", "threshold": "Threshold", "json_schema": "Fields", "pages": "Pages",
        "table_format": "Tables", "target_height": "Height", "target_fps": "Frame rate", "factor": "Factor", "texture_size": "Texture", "simplify": "Simplify",
        "multimask": "Show alternatives", "word_timestamps": "Word times", "diarize": "Speakers", "min_speakers": "At least", "max_speakers": "At most",
        "top_n": "Top", "include_image_base64": "Images", "intrinsics": "Camera parameters", "points": "Points", "boxes": "Boxes", "mode": "Mode",
        "normalize": "Normalize", "truncate": "Truncate", "dimensions": "Dimensions", "task": "Task", "phrase": "Phrase", "context": "Context",
        "start_frame": "From frame", "end_frame": "To frame", "max_tokens_per_doc": "Passage limit", "focal_length": "Focal length",
        // Translate's rules and preferences (the makers' hard and soft constraints).
        "hard_constraints": "Rules", "soft_constraints": "Preferences"
    ]

    /// `width` and `height` with bounds make one free size (and `resolution` is not shown); else a `resolution` list.
    private static func size(_ properties: [String: JSONObject]) -> (key: String, covers: [String], sizes: ImageSizes)? {
        func range(_ key: String) -> (Int, Int)? {
            guard let p = properties[key], let lo = p["minimum"]?.number?.roundedInt, let hi = p["maximum"]?.number?.roundedInt else { return nil }
            return (lo, hi)
        }
        if let (lo, hi) = range("width"), let (lo2, hi2) = range("height") {
            // A schema can say anything (a local folder): reversed bounds offer no size rather than trap.
            guard lo <= hi, lo2 <= hi2 else {
                return ("size", ["width", "height", "resolution"], ImageSizes(sizes: [], free: true))
            }
            let sizes = ImageSize.free.filter { (lo...hi).contains($0.width) && (lo2...hi2).contains($0.height) }
            return ("size", ["width", "height", "resolution"], ImageSizes(sizes: sizes, free: true, widthRange: lo...hi, heightRange: lo2...hi2))
        }
        // A list of "WxH" strings only: any other `resolution` (a number, a word) is the model's own option.
        if let choices = properties["resolution"]?["enum"]?.array, choices.count > 1 {
            let sizes = choices.compactMap { $0.string.flatMap(ImageSize.init(resolution:)) }
            if sizes.count == choices.count { return ("resolution", ["resolution"], ImageSizes(sizes: sizes, free: false)) }
        }
        return nil
    }

    static func control(_ key: String, _ p: JSONObject, root: SchemaRefs, operation: String, output: String) -> ToolField.Control {
        let types = p["type"]?.array?.compactMap(\.string).filter { $0 != "null" } ?? [p["type"]?.string].compactMap { $0 }
        let string = types == ["string"] && p["enum"] == nil
        let items = p["items"]?.object.map(root.resolve)
        let itemTypes = items?["type"]?.array?.compactMap(\.string) ?? [items?["type"]?.string].compactMap { $0 }
        // The folder's semantic hint first: it names the control the property means.
        switch p["x-arsenal-control"]?.string {
        case "language" where types == ["string"]:
            let choices = p["enum"]?.array?.compactMap(\.string)
            return .language(choices: choices.flatMap { $0.isEmpty ? nil : $0 })
        case "lyrics" where string: return .lyrics
        case "json-schema": return .schema
        // An object a document's annotations follow is a JSON Schema too, edited as named fields.
        case nil where key == "document_annotation_format" && types == ["object"]: return .schema
        case "image-points" where types == ["array"]: return .points(image: p["x-arsenal-image"]?.string ?? "image")
        case "image-boxes" where types == ["array"]: return .boxes(image: p["x-arsenal-image"]?.string ?? "image")
        default: break
        }
        if types == ["array"], !imageKeys.contains(key), OperationForm.fileAccept(p) == nil, items.flatMap(OperationForm.fileAccept) == nil {
            // Spatial prompts a folder left unannotated keep their meaning by shape (track_video's boxes).
            if key == "boxes", items?["properties"]?["xmin"] != nil { return .boxes(image: p["x-arsenal-image"]?.string ?? "video") }
            if key == "points", items?["properties"]?["x"] != nil, items?["properties"]?["y"] != nil { return .points(image: p["x-arsenal-image"]?.string ?? "image") }
            if itemTypes == ["string"], items?["enum"] == nil {
                return ["documents", "input", "text", "texts"].contains(key) ? .lines : .tags
            }
            if itemTypes == ["integer"], key == "pages" { return .pages }
            if itemTypes == ["number"] { return .numbers(min: items?["minimum"]?.number, max: items?["maximum"]?.number) }
        }
        switch key {
        case "prompt" where string, "text" where string:
            let placeholder =
                key == "text"
                ? "What should it say"
                : operation.contains("music")
                    ? "Describe the music"
                    : output == "image"
                        ? "Describe the image"
                        : output == "audio" || operation.contains("sound") ? "Describe the sound" : p["title"]?.string ?? OperationForm.title(key)
            return .main(placeholder: placeholder)
        case "voice" where string: return .voice
        case "seed" where types == ["integer"]: return .seed
        case "duration" where types == ["number"] || types == ["integer"]:
            let lo = p["minimum"]?.number ?? p["exclusiveMinimum"]?.number.map { $0 > 0 ? $0 : 0 } ?? 0
            let hi = p["maximum"]?.number ?? p["exclusiveMaximum"]?.number ?? 600
            if lo.isFinite, hi.isFinite, lo < hi { return .seconds(min: lo, max: hi) }
        case "instructions" where string, "description" where string, "negative_prompt" where string: return .note
        case "language" where string, "source_lang" where string, "target_lang" where string: return .language(choices: nil)
        case "reference" where types == ["object"] || p["properties"] != nil:
            if let inner = p["properties"]?.object, inner["path"] != nil {
                let needsText = p["required"]?.array?.contains(.string("text")) == true
                return .reference(transcript: inner["text"] == nil ? nil : needsText)
            }
        default:
            // Reference images sent inline: an array that is not a list of files (those are the generic file well's).
            if imageKeys.contains(key), types == ["array"] || p["maxItems"] != nil, OperationForm.fileAccept(p) == nil,
                p["items"]?.object.flatMap(OperationForm.fileAccept) == nil
            {
                return .images(max: max(0, min(16, p["maxItems"]?.number?.roundedInt ?? 4)))
            }
        }
        return .generic(OperationForm.control(p, root: root))
    }

    /// The generic form over the fields it draws, so their values encode exactly as that form encodes them.
    private var generic: OperationForm {
        var form = OperationForm(schema: ["type": "object", "properties": [:]])
        form.wholeJSON = wholeJSON
        form.fields = fields.compactMap { f in
            guard case .generic(let c) = f.control else { return nil }
            return FormField(key: f.key, title: f.title, help: f.help, control: c, required: f.required, defaultValue: f.defaultValue)
        }
        return form
    }

    /// The value a field starts with: the model's own setting of that name, else the schema's default.
    func initialValue(_ f: ToolField, setting: JSONValue?) -> FieldValue {
        let seed = setting ?? f.defaultValue
        switch f.control {
        case .main, .note, .voice: return .text(seed?.string ?? "")
        case .seed, .seconds: return .text(seed?.number.map(OperationForm.number) ?? "")
        case .reference, .images: return .files([])
        case .size(let s): return .text(seed?.string.flatMap(ImageSize.init(resolution:)).map(\.resolution) ?? s.preferred?.resolution ?? "")
        case .language, .lyrics: return .text(seed?.string ?? "")
        case .tags: return .text(seed?.array?.compactMap(\.string).joined(separator: ", ") ?? "")
        case .lines: return .text(seed?.array?.compactMap(\.string).joined(separator: "\n") ?? "")
        case .pages: return .text(seed?.array?.compactMap { $0.number.map { OperationForm.number($0 + 1) } }.joined(separator: ", ") ?? "")
        case .numbers: return .text(seed?.array?.compactMap { $0.number.map(OperationForm.number) }.joined(separator: ", ") ?? "")
        case .schema, .points, .boxes: return .text(seed.flatMap(OperationForm.pretty) ?? "")
        case .generic(let c):
            return generic.initialValue(FormField(key: f.key, title: f.title, help: f.help, control: c, required: f.required, defaultValue: f.defaultValue), setting: setting)
        }
    }

    /// The value from a model's settings: a size from its `width` and `height` (or `resolution`), else the setting of the
    /// field's own name; nil when the settings say nothing about it.
    func initialValue(_ f: ToolField, settings: [String: JSONValue]) -> FieldValue? {
        if case .size(let s) = f.control {
            if s.reference { return nil } // an edit keeps its reference's size unless the window picks one
            if s.free, let w = settings["width"]?.number?.roundedInt, let h = settings["height"]?.number?.roundedInt, s.accepts(ImageSize(width: w, height: h)) {
                return .text(ImageSize(width: w, height: h).resolution)
            }
            if let r = settings["resolution"]?.string.flatMap(ImageSize.init(resolution:)), s.accepts(r) || s.sizes.contains(r) { return .text(r.resolution) }
            return nil
        }
        return settings[f.key].map { initialValue(f, setting: $0) }
    }

    var main: ToolField? { fields.first(where: \.isMain) }

    /// "a, b\nc" → ["a", "b", "c"]: trimmed, empty parts dropped.
    static func items(_ text: String, separators: String) -> [String] {
        text.split(whereSeparator: { separators.contains($0) }).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
    /// "1-3, 5" → [0, 1, 2, 4] (pages as people count them, sent zero-based); nil when unreadable.
    static func pages(_ text: String) -> [Int]? {
        var pages: [Int] = []
        for part in items(text, separators: ",; ") {
            let bounds = part.split(whereSeparator: { $0 == "-" || $0 == "\u{2013}" }).map { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard let first = bounds.first ?? nil, first >= 1, bounds.count <= 2 else { return nil }
            let last = bounds.count == 2 ? bounds[1] : first
            guard let last, last >= first, last - first < 10_000 else { return nil }
            for page in first...last where !pages.contains(page - 1) { pages.append(page - 1) }
        }
        return pages.isEmpty ? nil : pages
    }
    /// The top-level `oneOf`/`anyOf` of required groups ("an image or a session"): each alternative's keys.
    static func alternatives(_ schema: JSONObject) -> [[String]] {
        let groups = (schema["oneOf"] ?? schema["anyOf"])?.array ?? []
        return groups.compactMap { $0.object?["required"]?.array?.compactMap(\.string) }.filter { !$0.isEmpty }
    }
    var takesSeed: Bool { fields.contains { $0.control == .seed } }

    /// The request body: every filled field; `seed` stands in for an empty seed field (a random seed the window keeps
    /// with its result); files are staged with `stage`.
    func input(_ values: [String: FieldValue], seed: Int?, stage: (URL) throws -> String) throws -> JSONObject {
        if wholeJSON { return try generic.input(values, stage: stage) }
        var input = try generic.input(values, stage: stage).merging(fixed) { a, _ in a }
        for f in fields {
            let value = values[f.key] ?? initialValue(f, setting: nil)
            var encoded: JSONValue?
            switch f.control {
            case .main where f.sendsList:
                let trimmed = value.text.trimmingCharacters(in: .whitespacesAndNewlines)
                encoded = trimmed.isEmpty ? nil : .array([.string(trimmed)])
            case .main, .note, .voice, .language:
                let trimmed = value.text.trimmingCharacters(in: .whitespacesAndNewlines)
                encoded = trimmed.isEmpty ? nil : .string(trimmed)
            case .lyrics:
                // An instrumental has no words: the lyrics stay in the window but are not sent.
                if case .toggle(true) = values["instrumental"] ?? .toggle(false) { encoded = nil; break }
                let trimmed = value.text.trimmingCharacters(in: .whitespacesAndNewlines)
                encoded = trimmed.isEmpty ? nil : .string(trimmed)
            case .tags:
                let items = Self.items(value.text, separators: ",\n")
                encoded = items.isEmpty ? nil : .array(items.map(JSONValue.string))
            case .lines:
                let items = Self.items(value.text, separators: "\n")
                encoded = items.isEmpty ? nil : .array(items.map(JSONValue.string))
            case .pages:
                let t = value.text.trimmingCharacters(in: .whitespaces)
                if t.isEmpty { break }
                guard let pages = Self.pages(t) else { throw OperationForm.Problem(key: f.key, message: "Type pages as 1-3, 5.") }
                encoded = .array(pages.map { .number(Double($0)) })
            case .numbers(let lo, let hi):
                let parts = Self.items(value.text, separators: ", \n")
                if parts.isEmpty { break }
                let numbers = parts.compactMap { Double($0) }
                guard numbers.count == parts.count, numbers.allSatisfy({ $0 >= (lo ?? -.infinity) && $0 <= (hi ?? .infinity) }) else {
                    let range = lo.flatMap { l in hi.map { h in " from \(OperationForm.number(l)) to \(OperationForm.number(h))" } } ?? ""
                    throw OperationForm.Problem(key: f.key, message: "\(f.title) takes numbers\(range), separated by commas.")
                }
                encoded = .array(numbers.map(JSONValue.number))
            case .schema, .points, .boxes:
                let t = value.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if t.isEmpty { break }
                guard let data = t.data(using: .utf8), let v = try? JSONDecoder().decode(JSONValue.self, from: data) else {
                    throw OperationForm.Problem(key: f.key, message: "\(f.title) is not valid JSON.")
                }
                if case .array(let list) = v, list.isEmpty { break }
                encoded = v
            case .seed:
                let t = value.text.trimmingCharacters(in: .whitespaces)
                if t.isEmpty {
                    encoded = seed.map { .number(Double($0)) }
                } else {
                    guard let n = Int(t), n >= 0 else { throw OperationForm.Problem(key: f.key, message: "The seed must be a whole number from 0.") }
                    encoded = .number(Double(n))
                }
            case .seconds(let lo, let hi):
                let t = value.text.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")
                if !t.isEmpty {
                    guard let n = Double(t), n.isFinite, n >= lo, n > 0, n <= hi else {
                        throw OperationForm.Problem(key: f.key, message: "\(f.title) must be more than \(OperationForm.number(lo)) and at most \(OperationForm.number(hi)) s.")
                    }
                    encoded = .number(n)
                }
            case .size(let s):
                guard let size = ImageSize(resolution: value.text) else { break }
                guard s.accepts(size) else { throw OperationForm.Problem(key: f.key, message: "This model does not take \(size.title).") }
                if s.free {
                    input["width"] = .number(Double(size.width)); input["height"] = .number(Double(size.height))
                } else {
                    input["resolution"] = .string(size.resolution)
                }
                continue
            case .images:
                let encodedImages = try value.files.map { url -> JSONValue in
                    do { return .string(try Data(contentsOf: url).base64EncodedString()) } catch {
                        throw OperationForm.Problem(key: f.key, message: "Couldn\u{2019}t read \(url.lastPathComponent).")
                    }
                }
                encoded = encodedImages.isEmpty ? nil : .array(encodedImages)
            case .reference(let transcript):
                guard let clip = value.files.first else { break }
                var object: JSONObject = ["path": .string(try stage(clip))]
                let words = values[Self.transcriptKey(f.key)]?.text.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                if transcript == true, words.isEmpty {
                    throw OperationForm.Problem(key: Self.transcriptKey(f.key), message: "Type what the clip says: this model needs its transcript.")
                }
                if transcript != nil, !words.isEmpty { object["text"] = .string(words) }
                encoded = .object(object)
            case .generic: continue
            }
            if let encoded {
                input[f.key] = encoded
            } else if f.required {
                if case .main = f.control { throw OperationForm.Problem(key: f.key, message: "Type something first.") }
                throw OperationForm.Problem(key: f.key, message: "\(f.title) is needed.")
            }
        }
        return input
    }
}
