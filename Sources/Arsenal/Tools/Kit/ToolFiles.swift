// SPDX-License-Identifier: MIT
import AppKit
import CryptoKit
import Foundation
import ImageIO
import ArsenalCore
import UniformTypeIdentifiers

// The tool kit's files: inputs staged in Arsenal's inbox (where the service accepts input files; an image the operation
// cannot take as it is goes as an upright PNG), an operation's result read through JSON negotiation (its files fetched
// from their job URLs, hash checked), results saved into the tool's folder under a dated name, and Copy and Show in
// Finder for whatever a window shows.

/// One file of a result, fetched to the window's own folder.
struct ResultFile: Equatable, Identifiable, Sendable {
    var id: String
    var label: String
    var order: Int
    var mime: String
    var url: URL
    var bytes: Int
    var width: Int?
    var height: Int?
    /// The runner's own fields for this file (depth_min 0.0, …), as "key value" pairs in a stable order.
    var extras: [Pair]
    var isImage: Bool { mime.hasPrefix("image/") }
    var isAudio: Bool { mime.hasPrefix("audio/") }
    /// "Near.png" (the label, made safe for a file name, with the type's extension).
    var suggestedName: String {
        let safe = ToolFiles.safe(label).trimmingCharacters(in: .whitespaces)
        return (safe.isEmpty ? id : safe) + "." + (url.pathExtension.isEmpty ? "bin" : url.pathExtension)
    }
}

struct Pair: Equatable, Hashable, Sendable {
    var key: String
    var value: String
    /// As a window shows it ("Mean depth", "1,048,576"); `key` and `value` stay raw for copied and exported data.
    var label: String { ResultWords.label(key) }
    var shown: String { ResultWords.value(value) }
    /// "Mean depth 0.4182", for a line of several.
    var line: String { label + " " + shown }
}

/// A reply's own keys and values, worded for a window: depth_mean → "Mean depth", depth_units → "Depth units",
/// 1048576 → "1,048,576". Raw keys stay in copied and exported data.
enum ResultWords {
    /// A statistic's suffix leads, as a person says it: depth_mean → "Mean depth", depth_max → "Max depth".
    static let leadingSuffixes = ["mean": "Mean", "min": "Min", "max": "Max", "avg": "Average", "median": "Median"]
    static func label(_ key: String) -> String {
        var words = key.replacingOccurrences(of: "-", with: "_").split(separator: "_").map(String.init).filter { !$0.isEmpty }
        guard !words.isEmpty else { return key }
        if words.count > 1, let lead = leadingSuffixes[words.last!.lowercased()] { words.removeLast(); words.insert(lead, at: 0) }
        // Sentence case, keeping an acronym as written (FPS, RMS).
        let first = words[0]
        words[0] = first == first.uppercased() && first.count > 1 ? first : first.prefix(1).uppercased() + first.dropFirst()
        for i in words.indices.dropFirst() where words[i] != words[i].uppercased() || words[i].count == 1 { words[i] = words[i].lowercased() }
        return words.joined(separator: " ")
    }
    /// A whole number grouped by thousands ("1,048,576"); anything else as the reply has it.
    static func value(_ v: String) -> String {
        guard let n = Int(v), abs(n) >= 1000 else { return v }
        let f = NumberFormatter()
        f.numberStyle = .decimal; f.groupingSeparator = ","; f.usesGroupingSeparator = true; f.groupingSize = 3
        return f.string(from: NSNumber(value: n)) ?? v
    }
}

struct OperationResult: Equatable, Sendable {
    var operation: String
    var files: [ResultFile] = []
    /// The reply without its files (metadata, or the whole object for a JSON operation).
    var fields: [Pair] = []
    /// A reply whose shape is not flat: shown as JSON.
    var json: String?
    var seconds: Double
    /// The whole reply without its file list, for windows that read its structure (segments, detections, pages).
    var reply: JSONObject = [:]
}

/// An image's upright size in pixels.
struct PixelSize: Equatable {
    var width: Int
    var height: Int
    /// "1920 × 1080".
    var title: String { "\(width) \u{00d7} \(height)" }

    /// The size as the image is meant to be seen (EXIF orientations 5–8 swap width and height); nil when unreadable.
    static func read(_ url: URL) -> PixelSize? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) > 0,
            let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let w = (props[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue, let h = (props[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue, w > 0, h > 0
        else { return nil }
        return (5...8).contains(orientation(source)) ? PixelSize(width: h, height: w) : PixelSize(width: w, height: h)
    }
    static func orientation(_ source: CGImageSource) -> Int {
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        return (props?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
    }
}

@MainActor enum ToolFiles {
    static var picturesDirectory: URL { kept(.picturesDirectory) }
    static var musicDirectory: URL { kept(.musicDirectory) }
    /// Where a result of this type is kept: images in Pictures, sound in Music, video in Movies, the rest (text, 3D,
    /// data) in Documents, each in its Arsenal folder. Tests set `override` to keep everything in their own folder.
    nonisolated(unsafe) static var override: URL?
    nonisolated static func folder(mime: String) -> URL {
        if let override { return override }
        let base: FileManager.SearchPathDirectory =
            mime.hasPrefix("image/") ? .picturesDirectory : mime.hasPrefix("audio/") ? .musicDirectory : mime.hasPrefix("video/") ? .moviesDirectory : .documentDirectory
        return kept(base)
    }
    /// `~/Pictures/Arsenal` and its siblings for the user's own profile. An isolated profile (`ARSENAL_SUPPORT_DIR` set:
    /// a lab, test or review copy) keeps its results inside itself, `<support>/Outputs/Pictures` and so on, so it never
    /// writes into the user's own folders.
    nonisolated static func kept(_ base: FileManager.SearchPathDirectory, environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let support = environment["ARSENAL_SUPPORT_DIR"], !support.isEmpty {
            let name: String
            switch base {
            case .picturesDirectory: name = "Pictures"
            case .musicDirectory: name = "Music"
            case .moviesDirectory: name = "Movies"
            default: name = "Documents"
            }
            return URL(fileURLWithPath: support, isDirectory: true).appendingPathComponent("Outputs", isDirectory: true).appendingPathComponent(name, isDirectory: true)
        }
        return FileManager.default.urls(for: base, in: .userDomainMask).first!.appendingPathComponent("Arsenal", isDirectory: true)
    }

    /// A name without path separators.
    nonisolated static func safe(_ name: String) -> String { name.map { "/:\\".contains($0) ? "-" : $0 }.reduce("") { $0 + String($1) } }

    nonisolated static func isImageFile(_ url: URL) -> Bool {
        url.isFileURL && (UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) ?? false)
    }

    // MARK: Inputs

    /// A copy in its own inbox folder (the service reads only files inside its inbox): an image for a field that takes
    /// images goes as it is when the field takes its type and it is upright, else as an upright PNG; any other file is
    /// copied (a clone on APFS).
    nonisolated static func stage(_ url: URL, accept: FileAccept, inbox: URL) throws -> (folder: URL, path: String) {
        let folder = inbox.appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            guard isImageFile(url), accept.mimes.allSatisfy({ $0.hasPrefix("image/") || $0 == "*/*" }) else {
                let name = url.lastPathComponent.hasPrefix(".") ? "input" + url.lastPathComponent : url.lastPathComponent
                let copy = folder.appendingPathComponent(name)
                try FileManager.default.copyItem(at: url, to: copy)
                return (folder, copy.path)
            }
            let stem = url.deletingPathExtension().lastPathComponent
            let safe = stem.isEmpty || stem.hasPrefix(".") ? "input" + stem : stem
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) > 0 else {
                throw ToolError("\(url.lastPathComponent) is not an image Arsenal can read.")
            }
            if accept.accepts(url), PixelSize.orientation(source) == 1 {
                let copy = folder.appendingPathComponent(safe + "." + url.pathExtension.lowercased())
                try FileManager.default.copyItem(at: url, to: copy)
                return (folder, copy.path)
            }
            guard accept.mimes.isEmpty || accept.mimes.contains("image/png") || accept.mimes.contains("image/*") else {
                throw ToolError("This model takes \(accept.title) images only.")
            }
            let png = folder.appendingPathComponent(safe + ".png")
            try upright(source, to: png)
            return (folder, png.path)
        } catch {
            try? FileManager.default.removeItem(at: folder)
            if let error = error as? ToolError { throw error }
            throw WordedError.file(error, verb: "read", name: url.lastPathComponent)
        }
    }

    /// The image as it is meant to be seen (its EXIF orientation applied), full size, as PNG.
    nonisolated static func upright(_ source: CGImageSource, to url: URL) throws {
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]
        let w = (props[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0, h = (props[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(1, max(w, h)), kCGImageSourceShouldCacheImmediately: true
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary),
            let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw ToolError("Could not convert the image to PNG.") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw ToolError("Could not convert the image to PNG.") }
    }

    // MARK: Results

    /// A fetched result file moved into the tool's folder as `<date> <time> <model> <tag>.<ext>`, numbered when taken.
    static func save(_ fetched: URL, into directory: URL, model: String, tag: String, ext: String? = nil, what: String) throws -> URL {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let ext = ext ?? (fetched.pathExtension.isEmpty ? "bin" : fetched.pathExtension)
            let base = [stamp.string(from: Date()), model, safe(tag)].filter { !$0.isEmpty }.joined(separator: " ")
            var url = directory.appendingPathComponent("\(base).\(ext)")
            var n = 2
            while FileManager.default.fileExists(atPath: url.path) { url = directory.appendingPathComponent("\(base) \(n).\(ext)"); n += 1 }
            try FileManager.default.moveItem(at: fetched, to: url)
            return url
        } catch {
            throw WordedError(line: "Couldn\u{2019}t save the \(what) in \(directory.lastPathComponent).", raw: error.localizedDescription)
        }
    }
    static let stamp: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd HH.mm.ss"; return f
    }()

    /// Fields the service adds or the window shows itself; the rest are the runner's own and are listed.
    nonisolated static let knownFileFields: Set<String> = ["id", "path", "url", "mime", "bytes", "sha256", "label", "order", "width", "height"]

    /// The operation's answer: a file's bytes (one file), several files as `multipart/mixed` (a schema-2 operation that
    /// made more than one), or the JSON reply with its files fetched into `folder`. `mime` is the whole Content-Type.
    /// Off the main thread: a result can be hundreds of megabytes to parse, hash and write.
    nonisolated static func result(op: String, data: Data, mime: String?, client: ArsenalClient, into folder: URL) async throws -> OperationResult {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let type = mime?.split(separator: ";").first.map { String($0).trimmingCharacters(in: .whitespaces).lowercased() }
        if type == "multipart/mixed", let boundary = mime.flatMap(Self.boundary) {
            let parts = Self.parts(data, boundary: boundary)
            guard !parts.isEmpty else { throw ToolError("The model answered with no file.") }
            let files = try parts.enumerated().map { i, part in try file(part.body, type: part.type ?? "application/octet-stream", id: "output-\(i)", order: i, into: folder) }
            return OperationResult(operation: op, files: files, seconds: 0)
        }
        guard type == nil || type == "application/json" || type?.hasSuffix("+json") == true else {
            // A file operation answered with its bytes (a single audio or image file).
            return OperationResult(operation: op, files: [try file(data, type: type ?? "application/octet-stream", id: "output", order: 0, into: folder)], seconds: 0)
        }
        let reply = try JSONDecoder().decode(JSONValue.self, from: data)
        guard case .object(var object) = reply else { return OperationResult(operation: op, json: OperationForm.pretty(reply), seconds: 0) }
        var files: [ResultFile] = []
        if let list = object["files"]?.array ?? object["images"]?.array, list.allSatisfy({ $0.object?["url"] != nil }), !list.isEmpty {
            // A single-file reply is its own file entry: its path is the file's, not a field.
            for key in ["files", "images", "path", "url", "mime", "bytes", "sha256"] { object[key] = nil }
            for (index, item) in list.enumerated() {
                guard let file = item.object else { continue }
                files.append(try await fetch(file, index: index, client: client, into: folder))
            }
            files.sort { ($0.order, $0.id) < ($1.order, $1.id) }
        }
        var result = OperationResult(operation: op, files: files, seconds: 0)
        result.reply = object
        if let metadata = object["metadata"]?.object, object.count == 1 { object = metadata }
        if object.values.allSatisfy(Self.isScalar), object.count <= 24 {
            result.fields = object.keys.sorted().map { Pair(key: $0, value: Self.scalar(object[$0]!)) }
        } else {
            result.json = OperationForm.pretty(.object(object))
        }
        return result
    }

    /// One file of an answer written into the window's folder.
    private nonisolated static func file(_ data: Data, type: String, id: String, order: Int, into folder: URL) throws -> ResultFile {
        let url = folder.appendingPathComponent("\(UUID().uuidString.prefix(8)).\(Self.ext(type))")
        try data.write(to: url, options: .atomic)
        let size = NSImage(data: data)?.representations.first.map { ($0.pixelsWide, $0.pixelsHigh) }
        return ResultFile(
            id: id, label: order == 0 && id == "output" ? "Output" : "Output \(order + 1)", order: order, mime: type, url: url, bytes: data.count, width: size?.0,
            height: size?.1, extras: [])
    }

    /// `boundary=…` of a multipart Content-Type.
    nonisolated static func boundary(_ contentType: String) -> String? {
        for parameter in contentType.split(separator: ";").dropFirst() {
            let pair = parameter.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            if pair.count == 2, pair[0].lowercased() == "boundary" { return pair[1].trimmingCharacters(in: CharacterSet(charactersIn: "\"")) }
        }
        return nil
    }

    /// The parts of a `multipart/mixed` body: each part's Content-Type and bytes.
    nonisolated static func parts(_ data: Data, boundary: String) -> [(type: String?, body: Data)] {
        let delimiter = Data("--\(boundary)".utf8), crlf = Data("\r\n".utf8), blank = Data("\r\n\r\n".utf8)
        var parts: [(String?, Data)] = []
        var cursor = data.startIndex
        guard let first = data.range(of: delimiter, in: cursor..<data.endIndex) else { return [] }
        cursor = first.upperBound
        while cursor < data.endIndex {
            // "--" after a delimiter closes the body.
            if data[cursor...].starts(with: Data("--".utf8)) { break }
            guard let headersEnd = data.range(of: blank, in: cursor..<data.endIndex),
                let next = data.range(of: crlf + delimiter, in: headersEnd.upperBound..<data.endIndex)
            else { break }
            let headers = String(decoding: data[cursor..<headersEnd.lowerBound], as: UTF8.self)
            let type = headers.split(separator: "\r\n").first { $0.lowercased().hasPrefix("content-type:") }.map {
                String($0.dropFirst("content-type:".count)).trimmingCharacters(in: .whitespaces)
            }
            parts.append((type, Data(data[headersEnd.upperBound..<next.lowerBound])))
            cursor = next.upperBound
        }
        return parts
    }

    private nonisolated static func fetch(_ file: JSONObject, index: Int, client: ArsenalClient, into folder: URL) async throws -> ResultFile {
        guard let address = file["url"]?.string else { throw ToolError("Arsenal returned a file without its address.") }
        let path = address.hasPrefix("http") ? URL(string: address).map { u in u.path + (u.query.map { "?" + $0 } ?? "") } ?? address : address
        let (data, response) = try await client.send("GET", path)
        if let expected = file["sha256"]?.string?.lowercased(), SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() != expected {
            throw ToolError("A result file arrived damaged (its hash does not match); run it again.")
        }
        let id = file["id"]?.string ?? "file-\(index)"
        let mime = file["mime"]?.string ?? response.mimeType ?? "application/octet-stream"
        let order = file["order"]?.number.flatMap { $0.isFinite && abs($0) < 1e9 ? Int($0) : nil } ?? index
        let safeID = String(id.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" ? $0 : "-" }.prefix(48))
        let url = folder.appendingPathComponent("\(UUID().uuidString.prefix(8))-\(safeID).\(ext(mime))")
        try data.write(to: url, options: .atomic)
        let extras = file.keys.filter { !knownFileFields.contains($0) && isScalar(file[$0]!) }.sorted(by: fieldOrder).map { Pair(key: $0, value: scalar(file[$0]!)) }
        return ResultFile(
            id: id, label: file["label"]?.string ?? OperationForm.title(id), order: order, mime: mime, url: url, bytes: data.count,
            width: file["width"]?.number.flatMap { Int(exactly: $0.rounded()) }, height: file["height"]?.number.flatMap { Int(exactly: $0.rounded()) }, extras: extras)
    }

    /// Field names in reading order: alphabetical, except that a `_min` comes before its `_max` ("depth_min 0 · depth_max 0.25").
    nonisolated static func fieldOrder(_ a: String, _ b: String) -> Bool {
        func key(_ s: String) -> String {
            s.hasSuffix("_min") ? String(s.dropLast(4)) + "_0" : s.hasSuffix("_max") ? String(s.dropLast(4)) + "_1" : s
        }
        return key(a) < key(b)
    }
    nonisolated static func isScalar(_ v: JSONValue) -> Bool {
        switch v {
        case .string, .number, .bool, .null: return true
        default: return false
        }
    }
    nonisolated static func scalar(_ v: JSONValue) -> String {
        switch v {
        case .string(let s): return s
        case .number(let n): return n.rounded() == n && abs(n) < 1e15 ? String(Int(n)) : String(format: "%.4g", n)
        case .bool(let b): return b ? "yes" : "no"
        default: return "\u{2014}"
        }
    }
    nonisolated static func ext(_ mime: String) -> String {
        switch mime.split(separator: ";").first.map(String.init) ?? mime {
        case "image/png": return "png"
        case "image/jpeg": return "jpg"
        case "image/webp": return "webp"
        case "audio/wav", "audio/x-wav", "audio/wave": return "wav"
        case "audio/flac": return "flac"
        case "audio/mpeg": return "mp3"
        case "application/json": return "json"
        case "text/plain": return "txt"
        case "application/x-npy": return "npy"
        case "video/mp4": return "mp4"
        case "video/quicktime": return "mov"
        case "model/gltf-binary": return "glb"
        case "model/gltf+json": return "gltf"
        case "model/vnd.usdz+zip": return "usdz"
        case "model/obj": return "obj"
        case "application/x-ply", "model/ply": return "ply"
        case "application/pdf": return "pdf"
        case "text/markdown": return "md"
        case "image/tiff": return "tiff"
        case "image/x-exr", "image/exr": return "exr"
        default: return "bin"
        }
    }

    // MARK: Copy and Show in Finder

    /// Copy a shown result: an image as PNG (transparency kept) and TIFF for apps that only take that; any file as its
    /// file URL, so Finder and other apps paste the file itself.
    static func copy(_ url: URL, to pasteboard: NSPasteboard) throws {
        let item = NSPasteboardItem()
        if isImageFile(url) {
            let data = try Data(contentsOf: url)
            if url.pathExtension.lowercased() == "png" { item.setData(data, forType: .png) }
            if let image = NSImage(data: data) {
                if url.pathExtension.lowercased() != "png", let tiff = image.tiffRepresentation,
                    let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:])
                {
                    item.setData(png, forType: .png)
                }
                if let tiff = image.tiffRepresentation { item.setData(tiff, forType: .tiff) }
            }
        } else {
            guard FileManager.default.fileExists(atPath: url.path) else { throw ToolError("\(url.lastPathComponent) is gone.") }
        }
        item.setString(url.absoluteString, forType: .fileURL)
        pasteboard.clearContents()
        pasteboard.writeObjects([item])
    }

    /// The shown file in Finder, else the tool's folder.
    static func reveal(_ url: URL?, folder: URL) {
        if let url {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } else {
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            NSWorkspace.shared.open(folder)
        }
    }
}
