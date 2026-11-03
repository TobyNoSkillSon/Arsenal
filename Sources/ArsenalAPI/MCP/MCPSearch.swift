// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// `arsenal_find` search: ranks models by what they do. A query word scores by the strongest thing it names in a model —
/// an operation or task it declares (directly or through a synonym: "invoice" → read_document) outranks its name,
/// languages, purpose, good-for notes, media, kind and summary. Limits are never searched: a "can't" line is evidence
/// against a model. Models that match only weakly are dropped instead of padding the list.
enum MCPSearch {
    enum Weight {
        static let operation = 10, name = 8, language = 6, purpose = 5, multilingual = 4, goodFor = 4, kind = 3, modality = 3, summary = 2
        /// A verb shared by unrelated operations ("remove" background, noise or vocals) says little on its own.
        static let weakVerb = 3
    }
    /// Hits scoring below this share of the best hit are dropped (only when the query names something specific).
    static let floor = 0.7

    /// A query split into words that name a task or subject and media words (speech, image, video…), stop words removed.
    struct Query: Equatable {
        var specific: [String], generic: [String]
        var isEmpty: Bool { specific.isEmpty && generic.isEmpty }
    }

    static let stop: Set<String> = [
        "to", "the", "a", "an", "of", "and", "or", "for", "in", "on", "with", "from", "into", "my", "me", "it", "is", "be", "find", "make", "create", "get", "use",
        "using", "want", "need", "some", "something", "this", "that", "these", "those", "can", "could", "would", "do", "does", "model", "models", "tool", "tools",
        "help", "please", "we", "you", "your", "our", "at", "by", "as", "are", "was", "what", "which", "how", "all", "any", "via", "up", "one"
    ]
    /// Media words: they narrow a search but never decide it ("transcribe speech" is about transcribing).
    static let modality: [String: String] = [
        "speech": "audio", "audio": "audio", "sound": "audio", "sounds": "audio", "recording": "audio", "recordings": "audio", "image": "image", "images": "image",
        "picture": "image", "pictures": "image", "photo": "image", "photos": "image", "video": "video", "videos": "video", "clip": "video", "clips": "video",
        "text": "text", "texts": "text", "file": "", "files": "", "data": "", "content": ""
    ]
    static let phrases: [(String, String)] = [
        ("speech to text", "transcribe"), ("text to speech", "speak"), ("read out loud", "speak"), ("read aloud", "speak"), ("out loud", "speak"),
        ("aloud", "speak"), ("who spoke", "diarize"), ("who is speaking", "diarize"), ("who speaks", "diarize"), ("frame rate", "framerate"),
        ("slow motion", "slowmotion"), ("in between", "inbetween"), ("speaker turns", "diarize"), ("identify speakers", "diarize"),
        ("identify speaker", "diarize"), ("meeting speakers", "diarize"), ("turn taking", "endpointing"), ("finished speaking", "endpointing"),
        ("how far", "distance"), ("remove sound", "separate"), ("remove the sound", "separate")
    ]
    static let weakVerbs = ["remove", "read", "estimate", "ground", "end", "design", "select", "separate"]
    /// Everyday words for an operation or task name, matched by stem.
    static let concepts: [String: Set<String>] = {
        let table: [([String], [String])] = [
            (["transcribe"], ["transcribe", "transcription", "transcript", "dictation", "dictate", "stt", "asr", "subtitle", "caption"]),
            (
                ["read_document"],
                ["read", "ocr", "scan", "scanned", "scanner", "document", "invoice", "receipt", "pdf", "page", "handwriting", "handwritten", "form", "paper", "formula"]
            ),
            (["extract"], ["invoice", "receipt", "form", "field", "json", "structured", "template", "extract"]),
            (["translate"], ["translate", "translation", "translator"]),
            (["speak"], ["speak", "say", "tts", "narrate", "narration", "voiceover", "pronounce", "voice"]),
            (["clone_voice"], ["clone", "cloning", "voice"]),
            (["design_voice"], ["voice"]),
            (["estimate_depth"], ["depth", "distance"]),
            (["image_to_3d"], ["3d", "mesh"]),
            (["remove_background"], ["background", "cutout", "matte", "transparent", "transparency"]),
            (["ground_screen"], ["button", "click", "screen", "ui", "gui", "element", "icon", "tap", "menu", "cursor", "locate"]),
            (["detect"], ["object", "detect", "detection", "box", "bounding", "locate", "count"]),
            (["segment"], ["segment", "segmentation", "mask", "select"]),
            (["track_video"], ["track", "tracking"]),
            (["separate_audio"], ["vocal", "stem", "karaoke", "acapella", "isolate", "separate", "separation"]),
            (["diarize"], ["diarize", "diarization", "speaker", "speakers"]),
            // Cleanup words name the declared denoise task, not the shared `enhance` op: AudioSR also runs `enhance`, but it
            // rebuilds high frequencies and adds a noise-like band, so it must not lead a noise query.
            (["enhance"], ["enhance"]),
            (["denoise"], ["denoise", "noise", "noisy", "cleanup", "clean", "echo", "reverb", "hiss", "hum"]),
            (["audio_super_resolution"], ["treble", "bandwidth", "muffled", "superresolution"]),
            (["upscale_image", "upscale_video"], ["upscale", "enlarge", "superresolution", "restore", "restoration", "sharpen", "blurry", "resolution"]),
            (["interpolate_video"], ["interpolate", "interpolation", "framerate", "fps", "slowmotion", "inbetween"]),
            (["colorize"], ["colorize", "colorise", "colorization", "colourize", "colourise", "colourisation", "monochrome", "grayscale", "greyscale", "sepia"]),
            (["inpaint"], ["inpaint", "inpainting", "erase", "eraser", "fill", "scratch", "scratches", "blemish", "watermark"]),
            (["generate_music"], ["music", "song", "lyrics", "singing", "melody", "instrumental"]),
            (["generate_sound"], ["sfx", "effect", "foley"]),
            (
                ["generate"],
                [
                    "draw", "drawing", "illustration", "illustrate", "paint", "painting", "art", "artwork", "hero", "banner", "backdrop", "wallpaper", "poster",
                    "sticker", "mockup"
                ]
            ),
            (["edit"], ["edit", "retouch", "inpaint", "inpainting"]),
            (["embed"], ["embed", "embedding", "vector", "semantic", "similarity", "similar", "search", "retrieval", "retrieve"]),
            (["rerank"], ["rerank", "rank", "ranking", "relevance", "search", "retrieval"]),
            (["decide", "judge"], ["decide", "decision", "classify", "classification", "triage", "judge", "categorize", "gate", "filter"]),
            (["end_of_turn"], ["endpoint", "endpointing", "interrupt", "interruption"])
        ]
        var concepts: [String: Set<String>] = [:]
        for (targets, keys) in table { for key in keys { concepts[stem(key), default: []].formUnion(targets) } }
        return concepts
    }()
    /// English language names to the codes model folders declare.
    static let languageNames: [String: String] = [
        "english": "en", "polish": "pl", "german": "de", "french": "fr", "spanish": "es", "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
        "ukrainian": "uk", "chinese": "zh", "mandarin": "zh", "cantonese": "yue", "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi", "czech": "cs",
        "swedish": "sv", "danish": "da", "finnish": "fi", "norwegian": "no", "greek": "el", "hungarian": "hu", "romanian": "ro", "bulgarian": "bg", "croatian": "hr",
        "slovak": "sk", "slovenian": "sl", "estonian": "et", "latvian": "lv", "lithuanian": "lt", "maltese": "mt", "turkish": "tr", "vietnamese": "vi", "thai": "th",
        "indonesian": "id", "malay": "ms", "filipino": "fil", "persian": "fa", "macedonian": "mk", "hebrew": "he"
    ]

    static func words(_ text: String) -> [String] {
        text.lowercased().split { !$0.isLetter && !$0.isNumber && $0 != "." }.map { $0.trimmingCharacters(in: CharacterSet(charactersIn: ".")) }.filter { $0.count >= 2 }
    }
    static func parse(_ text: String) -> Query {
        var joined = " " + words(text).joined(separator: " ") + " "
        for (phrase, replacement) in phrases { joined = joined.replacingOccurrences(of: " \(phrase) ", with: " \(replacement) ") }
        var tokens = joined.split(separator: " ").map(String.init)
        if tokens.contains("speak") { tokens.removeAll { $0 == "read" } }
        var seen: [String] = []
        for token in tokens where !stop.contains(token) && !seen.contains(token) { seen.append(token) }
        return Query(specific: seen.filter { modality[$0] == nil }, generic: seen.filter { modality[$0] != nil })
    }
    /// A light suffix strip, so "translation", "translates" and "translate" meet.
    static func stem(_ word: String) -> String {
        for suffix in ["ations", "ation", "ments", "ment", "ings", "ing", "ions", "ion", "ies", "es", "ed", "s", "e"]
        where word.hasSuffix(suffix) && word.count - suffix.count >= 4 {
            return String(word.dropLast(suffix.count)) + (suffix == "ies" ? "y" : "")
        }
        return word
    }
    /// Same word up to inflection: equal stems, a stem that extends the other by at most three letters ("remov" →
    /// "removal"), or a shared prefix of seven letters covering most of the shorter ("transcrib" ↔ "transcript").
    /// "translate" and "transparent" do not meet.
    static func similar(_ a: String, _ b: String) -> Bool {
        if a == b { return true }
        let x = stem(a), y = stem(b)
        if x == y { return true }
        let (short, long) = x.count <= y.count ? (x, y) : (y, x)
        if short.count >= 4 && long.hasPrefix(short) && long.count - short.count <= 3 { return true }
        let common = zip(short, long).prefix { $0 == $1 }.count
        return common >= 7 && Double(common) >= 0.8 * Double(short.count)
    }

    /// One model's searchable words, by field.
    struct Fields {
        var names: Set<String>, operationWords: Set<String>, name: Set<String>, purpose: Set<String>, goodFor: Set<String>, kind: Set<String>, summary: Set<String>
        var languages: [String], modalities: Set<String>
        init(_ entry: CatalogEntry, operations: [String]) {
            names = Set((operations + entry.tasks).map(MCPSearch.normalized))
            operationWords = Set(
                operations.flatMap { $0.lowercased().split(separator: "_").map(String.init) }.filter { $0.count >= 2 && !stop.contains($0) && modality[$0] == nil })
            name = Set(words(entry.id.replacingOccurrences(of: "-", with: " ") + " " + entry.name))
            purpose = Set(words(entry.row.purpose)); goodFor = Set(words(entry.goodFor.joined(separator: " "))); kind = Set(words(entry.kind))
            summary = Set(words([entry.summary, entry.description, entry.publisher].joined(separator: " ")))
            languages = entry.languages.map { $0.lowercased() }; modalities = Set(entry.modalitiesIn + entry.modalitiesOut)
        }
    }
    static func normalized(_ name: String) -> String { name.lowercased().replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: " ", with: "_") }

    struct WordMatch { var score = 0, operation = false }
    /// The word's best field weight, plus one for each further field it also matches.
    static func match(_ word: String, _ fields: Fields) -> WordMatch {
        var weights: [Int] = [], operation = false
        if !(concepts[stem(word)] ?? []).isDisjoint(with: fields.names) {
            weights.append(Weight.operation); operation = true
        } else if fields.operationWords.contains(where: { similar(word, $0) }) {
            if weakVerbs.contains(where: { similar(word, $0) }) { return WordMatch(score: Weight.weakVerb) }
            weights.append(Weight.operation); operation = true
        }
        if let code = languageNames[word] {
            switch languageFit(fields.languages, code: code) {
            case .exact: weights.append(Weight.language)
            case .multilingual: weights.append(Weight.multilingual)
            case .none: break
            }
        }
        if let medium = modality[word], fields.modalities.contains(medium) { weights.append(Weight.modality) }
        let fieldWeights = [
            (fields.name, Weight.name), (fields.purpose, Weight.purpose), (fields.goodFor, Weight.goodFor), (fields.kind, Weight.kind), (fields.summary, Weight.summary)
        ]
        for (field, weight) in fieldWeights where field.contains(where: { similar(word, $0) }) { weights.append(weight) }
        guard let best = weights.max() else { return WordMatch() }
        return WordMatch(score: best + weights.count - 1, operation: operation)
    }

    /// Relevance scores of the models that fit the query; models missing from the result do not fit.
    /// When some model does what the query asks (an operation match), only such models stay; otherwise a model must
    /// match at least half the specific words. Then anything under `floor` of the best score for its lead word is dropped.
    static func relevant(_ entries: [CatalogEntry], operations: [String: [String]], query text: String) -> [String: Int] {
        let query = parse(text)
        guard !query.isEmpty else { return [:] }
        var hits: [(id: String, score: Int, operation: Bool, specific: Int, lead: String)] = []
        for entry in entries {
            let fields = Fields(entry, operations: operations[entry.id] ?? [])
            var hit = (id: entry.id, score: 0, operation: false, specific: 0, lead: "")
            var leadScore = 0
            for word in query.specific + query.generic {
                let found = match(word, fields)
                guard found.score > 0 else { continue }
                hit.score += found.score; hit.operation = hit.operation || found.operation
                if query.specific.contains(word) {
                    hit.specific += 1
                    if found.score > leadScore { leadScore = found.score; hit.lead = word }
                }
            }
            if hit.score > 0 { hits.append(hit) }
        }
        if hits.contains(where: \.operation) {
            hits = hits.filter(\.operation)
        } else if !query.specific.isEmpty {
            let needed = (query.specific.count + 1) / 2
            hits = hits.filter { $0.specific >= needed }
        }
        if !query.specific.isEmpty {
            // Each hit is judged against the best hit whose strongest specific word is the same, so a query that names two
            // tasks ("speaker … voice") keeps the best of each rather than only the stronger one.
            var best: [String: Int] = [:]
            for hit in hits { best[hit.lead] = max(best[hit.lead] ?? 0, hit.score) }
            hits = hits.filter { Double($0.score) >= floor * Double(best[$0.lead] ?? 0) }
        }
        return Dictionary(uniqueKeysWithValues: hits.map { ($0.id, $0.score) })
    }

    /// Whether `task` names something the model does: one of its operations or tasks, its kind ("image" matches kind
    /// "images"), or everyday words for them ("speech to text", "ocr", "depth").
    static func matches(_ entry: CatalogEntry, operations: [String], task: String) -> Bool {
        let wanted = normalized(task.trimmingCharacters(in: .whitespacesAndNewlines))
        let names = Set((operations + entry.tasks).map(normalized) + [entry.kind.lowercased()])
        if names.contains(wanted) || names.contains(wanted + "s") || (wanted.hasSuffix("s") && names.contains(String(wanted.dropLast()))) { return true }
        if names.contains(where: { $0.hasPrefix(wanted + "_") || $0.hasSuffix("_" + wanted) }) { return true }
        let query = parse(task)
        guard !query.isEmpty else { return false }
        let fields = Fields(entry, operations: operations)
        for word in query.specific where !match(word, fields).operation && !fields.kind.contains(where: { similar(word, $0) }) { return false }
        for word in query.generic where match(word, fields).score == 0 { return false }
        return true
    }

    enum LanguageFit { case none, multilingual, exact }
    /// A language filter or query word as a code: "Polish", "pl" and "pl-PL" are all "pl".
    static func languageCode(_ text: String) -> String {
        let lower = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let code = languageNames[lower] { return code }
        return String(lower.split { $0 == "-" || $0 == "_" }.first ?? Substring(lower))
    }
    /// Exact when the model declares the language; multilingual when it declares many ("mul") without listing them.
    static func languageFit(_ languages: [String], code: String) -> LanguageFit {
        if languages.contains(where: { $0.lowercased() == code || $0.lowercased().hasPrefix(code + "-") }) { return .exact }
        return languages.contains { $0.lowercased() == "mul" } ? .multilingual : .none
    }

    static func distance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        guard !a.isEmpty else { return b.count }
        guard !b.isEmpty else { return a.count }
        var row = Array(0...b.count)
        for i in 1...a.count {
            var previous = row[0]; row[0] = i
            for j in 1...b.count {
                let current = row[j]
                row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] == b[j - 1] ? 0 : 1))
                previous = current
            }
        }
        return row[b.count]
    }
}
