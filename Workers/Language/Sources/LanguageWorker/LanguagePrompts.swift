// SPDX-License-Identifier: MIT
import Foundation
import MLXLMCommon
import ArsenalWire

enum LanguagePrompts {
    /// Index-Translate inference_llm_call_api.py image prompt; extras keep the maker's labels.
    static func imageTranslation(text: String, target: String, input: JSONObject) throws -> String {
        let targetName = try indexLanguage(target)
        let sourceName = try indexSource(input)
        let base = "请将图片中的\(sourceName)文本翻译为\(targetName)，直接输出翻译结果，不要进行任何解释。"
        var extras: [String] = []
        if !text.isEmpty { extras.append("【参考文本/关注内容】\n" + text) }
        for key in ["instructions", "tone", "context"] { if let value = input[key]?.string, !value.isEmpty { extras.append("【指令要求】" + value) } }
        if let format = input["format"]?.string { extras.append("【指令要求】" + indexFormat(format)) }
        extras += indexLines(input["hard_constraints"], prefix: "【硬性要求】", keep: ["【硬性要求】"])
        extras += indexLines(input["soft_constraints"], prefix: "【注意】", keep: ["【注意】", "【软性要求】"])
        let terms = try indexGlossary(input)
        if !terms.isEmpty { extras.append("【术语对照】" + terms.joined(separator: "、")) }
        return extras.isEmpty ? base : base + "\n\n" + extras.joined(separator: "\n\n") + "\n\n只输出翻译结果，不要进行任何解释。"
    }
    static func messages(prompt: String, system: String?) -> [Message] {
        var result: [Message] = []
        if let system, !system.isEmpty { result.append(["role": "system", "content": system]) }
        result.append(["role": "user", "content": [["type": "image"], ["type": "text", "text": prompt]]])
        return result
    }
    /// Index-Translate image requests put the instruction text before the image (inference_llm_call_api.py).
    static func textThenImage(prompt: String) -> [Message] {
        [["role": "user", "content": [["type": "text", "text": prompt], ["type": "image"]]]]
    }
    /// Index-Translate inference_llm_translate.py trans_prompt.
    static func translation(id: String, text: String, target: String, input: JSONObject) throws -> String {
        if id == "hy-mt2-7b" { return try hunyuan(text: text, target: target, input: input) }
        let targetName = try indexLanguage(target)
        let sourceName = try indexSource(input)
        let stripped = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // Order as trans_prompt: hard constraints (the format rule is one), glossary, soft constraints, instructions.
        var constraints = indexLines(input["hard_constraints"], prefix: "【硬性要求】", keep: ["【硬性要求】"])
        if let format = input["format"]?.string { constraints.append("【硬性要求】" + indexFormat(format)) }
        let terms = try indexGlossary(input)
        if !terms.isEmpty { constraints.append("【硬性要求】专名/术语对照: " + terms.joined(separator: "、")) }
        constraints += indexLines(input["soft_constraints"], prefix: "【注意】", keep: ["【注意】", "【软性要求】"])
        for key in ["tone", "context", "instructions"] {
            constraints += indexLines(input[key], prefix: "【注意】", keep: ["【硬性要求】", "【注意】", "【软性要求】"])
        }
        if constraints.isEmpty { return "请将以下\(sourceName)文本翻译为\(targetName)，直接输出翻译结果，不要进行任何解释。\n\n\(stripped)" }
        let genre = input["genre"]?.string.flatMap { $0.isEmpty ? nil : $0 } ?? "文本"
        let isObject =
            stripped.hasPrefix("{") && stripped.hasSuffix("}")
            && ((try? JSONSerialization.jsonObject(with: Data(stripped.utf8))) as? [String: Any]) != nil
        let suffix =
            isObject && constraints.contains(where: { $0.contains("JSON") || $0.contains("json") })
            ? "请以相同的 JSON 格式输出翻译结果，key 保持不变，value 为对应译文。只输出 JSON，不要有任何额外说明。" : "只输出译文，不要有任何额外说明。"
        return "请将以下\(sourceName)\(genre)翻译成\(targetName)，并且严格遵循所有约束要求。\n\n【源文】\n\(stripped)\n\n【约束要求】\n"
            + constraints.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n") + "\n\n" + suffix
    }
    /// trans_prompt line normalization: split lines, strip, drop leading numbering (`lstrip("0123456789. ")`),
    /// skip empty lines and add the prefix unless the line already carries an accepted one.
    static func indexLines(_ value: ArsenalWire.JSONValue?, prefix: String, keep: [String]) -> [String] {
        let items = value?.array?.compactMap(\.string) ?? value?.string.map { [$0] } ?? []
        return items.flatMap { $0.components(separatedBy: .newlines) }.compactMap { raw -> String? in
            let line = String(raw.trimmingCharacters(in: .whitespaces).drop(while: { "0123456789. ".contains($0) }))
            guard !line.isEmpty else { return nil }
            return keep.contains(where: { line.hasPrefix($0) }) ? line : prefix + line
        }
    }
    /// parse_glossary_terms for a JSON object: `key.strip()→value.strip()`.
    static func indexGlossary(_ input: JSONObject) throws -> [String] {
        guard let glossary = input["glossary"]?.object else { return [] }
        return try glossary.keys.sorted().map { key in
            guard let value = glossary[key]?.string else { throw WorkerFailure(code: "invalid_request", message: "Glossary values must be strings.") }
            return key.trimmingCharacters(in: .whitespaces) + "→" + value.trimmingCharacters(in: .whitespaces)
        }
    }
    /// The schema's format choice as a Chinese hard constraint; JSON keeps the word JSON for trans_prompt's suffix rule.
    static func indexFormat(_ format: String) -> String {
        switch format.lowercased() {
        case "json": return "仅翻译键值内容，严格保留合法 JSON 语法格式与键名"
        case "csv": return "严格保留 CSV 格式，分隔符、列数与行结构保持不变"
        case "markdown": return "严格保留 Markdown 格式标记与文档结构"
        case "code": return "仅翻译代码中的注释与面向用户的字符串，代码本身保持不变"
        default: return "严格保留原文的 \(format) 格式"
        }
    }
    /// Index-Translate strip_think: drop a `<think>…</think>` block if the model emits one.
    static func indexAnswer(_ text: String) -> String {
        var value = text
        if let close = value.range(of: "</think>") { value = String(value[close.upperBound...]) }
        value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.hasPrefix("<think>") { value = String(value.dropFirst("<think>".count)) }
        return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    /// Source language name, empty for auto (any case) or absent.
    static func indexSource(_ input: JSONObject) throws -> String {
        guard let source = input["source_lang"]?.string?.trimmingCharacters(in: .whitespaces), !source.isEmpty, source.lowercased() != "auto" else { return "" }
        return try indexLanguage(source)
    }
    /// Maker LANG_NAMES (Chinese names) for the codes it lists, with the picker's ISO 639-1 aliases mapped onto them;
    /// other languages use their full Chinese name. Region and script suffixes are dropped except for Chinese script.
    static func indexLanguage(_ value: String) throws -> String {
        let raw = value.trimmingCharacters(in: .whitespaces).lowercased()
        if let name = languages[raw] { return name }
        let code = raw.replacingOccurrences(of: "_", with: "-")
        if ["zh-hant", "zh-tw", "zh-hk", "zh-mo"].contains(code) || code.hasPrefix("zh-hant-") { return "繁体中文" }
        let base = String(code.split(separator: "-").first ?? "")
        if let name = languages[base] ?? indexAliases[base].flatMap({ languages[$0] }) { return name }
        if base.count >= 2, base.count <= 3, base.allSatisfy(\.isLetter),
            let name = Locale(identifier: "zh-Hans").localizedString(forLanguageCode: base), !name.isEmpty, name.lowercased() != base
        {
            return name
        }
        // A full language name (for example 英语 or English) is passed as written, as the maker's prompt docs allow.
        if raw.count > 3, !raw.contains(where: { $0.isNumber }) { return value.trimmingCharacters(in: .whitespaces) }
        throw WorkerFailure(code: "invalid_request", message: "Unknown language code: \(value).")
    }
    static let indexAliases = [
        "uk": "ukr_cyrl", "fa": "fas_arab", "cs": "ces_latn", "el": "ell_grek", "da": "dan_latn", "hu": "hun_latn", "fi": "fin_latn", "no": "nob_latn", "nb": "nob_latn",
        "sk": "slk_latn", "bg": "bul_cyrl", "tl": "fil"
    ]
    /// Hy-MT2 card "Translation Task Instruction Examples" (English prompts; one prompt type per request).
    static func hunyuan(text: String, target: String, input: JSONObject) throws -> String {
        let language = try hunyuanLanguage(target)
        let glossary = input["glossary"]?.object ?? [:]
        var kinds: [String] = glossary.isEmpty ? [] : ["glossary"]
        for key in ["tone", "context", "format", "instructions"] where !(input[key]?.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            kinds.append(key)
        }
        if input["preserve_delimiters"]?.bool == true { kinds.append("preserve_delimiters") }
        guard kinds.count <= 1 else {
            throw WorkerFailure(
                code: "invalid_request", message: "Hy-MT2 takes one of glossary, tone, context, format, instructions or preserve_delimiters per request.")
        }
        let value = { (key: String) in input[key]?.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "" }
        switch kinds.first ?? "" {
        case "glossary":
            let terms = try glossary.keys.sorted().map { key -> String in
                guard let term = glossary[key]?.string else { throw WorkerFailure(code: "invalid_request", message: "Glossary values must be strings.") }
                return "\(key) translates to \(term)"
            }
            return "Reference the following translations:\n" + terms.joined(separator: "\n")
                + "\n\nTranslate the following text into \(language). Note that you must ONLY output the translated result without any additional explanation:\n\n\(text)"
        case "tone":
            return "Please translate the following text into \(language). Note that the translation style must strictly conform to [\(value("tone"))]:\n\n\(text)"
        case "context":
            return
                "[Background Information]\n\(value("context"))\n\nPlease translate the following text into \(language), taking the provided background information into consideration.\n\n[Source Text]\n\(text)"
        case "format":
            let names = ["json": "JSON", "xml": "XML", "yaml": "YAML", "html": "HTML", "markdown": "Markdown", "csv": "CSV"]
            let format = names[value("format").lowercased()] ?? value("format")
            return "### Task\nTranslate the user-facing text within the following \(format) data into \(language).\n\n### Strict Rules\n"
                + "1. Structure Preservation: You MUST preserve the original \(format) data structure, nesting, hierarchy, and indentation exactly as they are.\n"
                + "2. Selective Translation: Translate ONLY the visible, user-facing text content/values.\n"
                + "3. Strict Non-Translation: NEVER translate or alter code tags, keys, properties, object names, or variable placeholders. "
                + "Leave them exactly in their original English/code form.\n\n"
                + "### Source Data\n\(text)"
        case "instructions":
            let preferences = value("instructions").components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            let tasks = preferences.enumerated().map { "\($0.offset + 1). \($0.element)" } + ["\(preferences.count + 1). Translate the [Source Text] into \(language)."]
            return "[Source Text]\n\(text)\n\n[Translation Tasks]\n" + tasks.joined(separator: "\n")
        case "preserve_delimiters":
            return "Please accurately translate the following text into \(language).\nYou must retain the exact same number of delimiters in the translation. "
                + "Strictly do not omit, escape, or translate these symbols, and pay close attention to their placement.\n\n\(text)"
        default:
            return "Translate the following text into \(language). Note that you should only output the translated result without any additional explanation:\n\n\(text)"
        }
    }
    static func document(id: String, input: JSONObject, spec: JSONObject) throws -> String {
        let mode = input["mode"]?.string ?? "ocr"
        if mode == "custom", let prompt = input["prompt"]?.string, !prompt.isEmpty { return prompt }
        guard let prompt = spec["prompts"]?.object?[mode]?.string else { throw WorkerFailure(code: "invalid_request", message: "Unsupported document mode: \(mode).") }
        return prompt
    }
    /// Hy-MT2 card "Supported Languages" (English names for the English prompts). Other codes are rejected.
    static let hunyuanLanguages = [
        "zh": "Chinese", "en": "English", "fr": "French", "pt": "Portuguese", "es": "Spanish", "ja": "Japanese", "tr": "Turkish", "ru": "Russian", "ar": "Arabic",
        "ko": "Korean", "th": "Thai", "it": "Italian", "de": "German", "vi": "Vietnamese", "ms": "Malay", "id": "Indonesian", "tl": "Filipino", "hi": "Hindi",
        "zh-hant": "Traditional Chinese", "pl": "Polish", "cs": "Czech", "nl": "Dutch", "km": "Khmer", "my": "Burmese", "fa": "Persian", "gu": "Gujarati", "ur": "Urdu",
        "te": "Telugu", "mr": "Marathi", "he": "Hebrew", "bn": "Bengali", "ta": "Tamil", "uk": "Ukrainian", "bo": "Tibetan", "kk": "Kazakh", "mn": "Mongolian",
        "ug": "Uyghur", "yue": "Cantonese"
    ]
    /// Card code or English name (any case); region suffixes are dropped except Chinese script and region variants.
    static func hunyuanLanguage(_ value: String) throws -> String {
        let code = value.trimmingCharacters(in: .whitespaces).lowercased().replacingOccurrences(of: "_", with: "-")
        if let name = hunyuanLanguages[code] { return name }
        if ["zh-tw", "zh-hk", "zh-mo"].contains(code) || code.hasPrefix("zh-hant-") { return "Traditional Chinese" }
        let base = String(code.split(separator: "-").first ?? "")
        let aliases = ["fil": "tl", "iw": "he"]
        if let name = hunyuanLanguages[aliases[base] ?? base] { return name }
        if let name = hunyuanLanguages.values.first(where: { $0.lowercased() == code }) { return name }
        throw WorkerFailure(code: "invalid_request", message: "Hy-MT2 does not translate into \(value); see the model's supported languages.")
    }
    static let languages = [
        "en": "英语", "zh": "中文", "de": "德语", "fr": "法语", "es": "西班牙语", "ja": "日语", "ko": "韩语", "pt": "葡萄牙语", "ru": "俄语", "ar": "阿拉伯语", "it": "意大利语", "nl": "荷兰语", "pl": "波兰语",
        "ro": "罗马尼亚语", "sv": "瑞典语", "tr": "土耳其语", "hi": "印地语", "vi": "越南语", "th": "泰语", "id": "印尼语", "ms": "马来语", "fil": "菲律宾语", "ukr_cyrl": "乌克兰语", "fas_arab": "波斯语",
        "ces_latn": "捷克语", "ell_grek": "希腊语", "dan_latn": "丹麦语", "hun_latn": "匈牙利语", "fin_latn": "芬兰语", "nob_latn": "书面挪威语", "slk_latn": "斯洛伐克语", "bul_cyrl": "保加利亚语"
    ]
}
