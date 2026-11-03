// SPDX-License-Identifier: MIT
// Request checks as the maker's pydantic schema makes them (OneJev qev/schema.py @81ce62f): extra fields are ignored,
// an instruction or option description ("Entry") is text, an object, an array or null, a choice has 1-255 options and a
// score 1-10 levels. Arsenal's decisions contract adds its 64-question bound.
import Foundation

public struct InvalidRequest: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

public enum Requests {
    public static let maxQuestions = 64

    static func entry(_ value: PyJSON?, _ what: String) throws -> PyJSON? {
        guard let value else { return nil }
        switch value {
        case .string, .object, .array: return value
        case .scalar("null"): return nil
        default: throw InvalidRequest("\(what) must be text, an object, an array or null.")
        }
    }

    /// Named questions in request order. A name sent twice keeps its first position and its last value, as a Python
    /// dict built from the JSON does.
    /// `batch`: judge's shorthand, a choice's criteria given as a plain list of labels (no descriptions).
    public static func questions(_ raw: PyJSON?, batch: Bool = false) throws -> [(String, Question)] {
        guard let fields = raw?.fields, !fields.isEmpty else { throw InvalidRequest("questions must contain at least one named question.") }
        let unique = dedupe(fields)
        guard unique.count <= maxQuestions else { throw InvalidRequest("questions must contain at most \(maxQuestions) named questions.") }
        return try unique.map { name, q in (name, try question(q, name: name, batch: batch)) }
    }

    /// Keys compare by their UTF-8 bytes, as Python's str keys do; Swift's String equality would merge canonically
    /// equivalent spellings ("é" precomposed and decomposed) that Python keeps apart.
    static func dedupe(_ fields: [(String, PyJSON)]) -> [(String, PyJSON)] {
        var order: [(String, [UInt8])] = [], values: [[UInt8]: PyJSON] = [:]
        for (k, v) in fields {
            let bytes = Array(k.utf8)
            if values[bytes] == nil { order.append((k, bytes)) }
            values[bytes] = v
        }
        return order.map { ($0.0, values[$0.1]!) }
    }

    public static func question(_ q: PyJSON, name: String, batch: Bool = false) throws -> Question {
        guard q.isObject, let type = q["type"]?.string, let kind = QuestionKind(rawValue: type) else {
            throw InvalidRequest("Question \(name) needs type noul, choice or score.")
        }
        let instructions = try entry(q["instructions"], "instructions")
        var criteria = q["criteria"]
        if batch, kind == .choice, let labels = criteria?.array {
            guard labels.allSatisfy({ $0.string != nil }) else { throw InvalidRequest("Choice label lists must contain strings.") }
            criteria = .object(labels.compactMap(\.string).map { ($0, PyJSON.scalar("null")) })
        }
        switch kind {
        case .noul:
            guard let c = criteria, !c.isNull else { return Question(kind: .noul, instructions: instructions, options: []) }
            guard let f = c.fields else { throw InvalidRequest("Noul criteria must be an object with optional true and false.") }
            let d = dedupe(f)
            let t = try entry(d.first { Array($0.0.utf8) == Array("true".utf8) }?.1, "criteria.true")
            let n = try entry(d.first { Array($0.0.utf8) == Array("false".utf8) }?.1, "criteria.false")
            return Question(kind: .noul, instructions: instructions, options: [("true", t), ("false", n)])
        case .choice:
            guard let f = criteria?.fields else { throw InvalidRequest("Choice question \(name) needs criteria: an object of named options.") }
            let options = try dedupe(f).map { ($0.0, try entry($0.1, "an option description")) }
            guard !options.isEmpty else { throw InvalidRequest("choice question needs at least one option in criteria") }
            guard options.count <= Prompt.maxChoiceOptions else {
                throw InvalidRequest("choice question has \(options.count) options, maximum is \(Prompt.maxChoiceOptions)")
            }
            return Question(kind: .choice, instructions: instructions, options: options)
        case .score:
            guard let levels = criteria?.array else { throw InvalidRequest("Score question \(name) needs criteria: a list of levels, lowest first.") }
            guard !levels.isEmpty else { throw InvalidRequest("score question needs at least one level in criteria") }
            guard levels.count <= Prompt.maxScoreLevels else {
                throw InvalidRequest("score question has \(levels.count) levels, maximum is \(Prompt.maxScoreLevels)")
            }
            return Question(kind: .score, instructions: instructions, options: try levels.map { ("", try entry($0, "a level")) })
        }
    }

    /// A media problem with its wire code: `invalid_request` for the request's shape (counts, kinds, fps, frame sizes),
    /// `invalid_image` for a file that is missing or cannot be read.
    public struct MediaError: Error, CustomStringConvertible {
        public let code: String
        public let description: String
    }

    /// The maker's `media` list (qev/media.py): images with a local file path, and videos as frame lists with an
    /// optional fps (default 1.0, as the makers'), in placeholder order. A frame is a path, as the makers send it, or a
    /// {"path": …} object, which Arsenal's file staging fills. Counts are checked before any file is touched; then only
    /// stored sizes are read, and pixels are decoded later, after the token limits pass.
    public static func media(_ raw: PyJSON?, maxItems: Int) throws -> [MediaItem] {
        guard let raw, !raw.isNull else { return [] }
        guard let items = raw.array, items.count <= maxItems else {
            throw MediaError(code: "invalid_request", description: "media must be a list of at most \(maxItems) items.")
        }
        func file(_ value: PyJSON?, _ what: String) throws -> URL {
            guard let path = value?.string, path.hasPrefix("/") else {
                throw MediaError(code: "invalid_image", description: "\(what) needs an absolute local path.")
            }
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), !isDirectory.boolValue else {
                throw MediaError(code: "invalid_image", description: "\(what): no file at that path.")
            }
            return URL(fileURLWithPath: path)
        }
        return try items.enumerated().map { index, item in
            guard item.isObject, let type = item["type"]?.string, ["image", "video"].contains(type) else {
                throw MediaError(code: "invalid_request", description: "media[\(index)] must be an object with type image or video.")
            }
            if type == "image" {
                let url = try file(item["path"], "media[\(index)]")
                do { return .image(try ImageInput.probe(url)) } catch { throw MediaError(code: "invalid_image", description: "media[\(index)]: \(error)") }
            }
            guard let frames = item["frames"]?.array, (2...VideoInput.maxFrames).contains(frames.count) else {
                throw MediaError(code: "invalid_request", description: "media[\(index)] (video) needs 2 to \(VideoInput.maxFrames) frames.")
            }
            var fps = 1.0
            if let f = item["fps"], !f.isNull {
                guard let value = Double(f.compact()), value > 0, value.isFinite else {
                    throw MediaError(code: "invalid_request", description: "media[\(index)].fps must be a positive number.")
                }
                fps = value
            }
            let urls = try frames.enumerated().map { try file($0.element.string != nil ? $0.element : $0.element["path"], "media[\(index)].frames[\($0.offset)]") }
            let sizes = try urls.enumerated().map { offset, url -> ImageSource in
                do { return try ImageInput.probe(url) } catch { throw MediaError(code: "invalid_image", description: "media[\(index)].frames[\(offset)]: \(error)") }
            }
            do { return .video(try VideoInput.probe(urls, fps: fps, sizes: sizes)) } catch {
                throw MediaError(code: "invalid_request", description: "media[\(index)]: \(error)")
            }
        }
    }
}
