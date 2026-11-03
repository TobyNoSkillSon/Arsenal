// SPDX-License-Identifier: MIT
import AppKit
import Foundation

/// The newest result any tool window made, for the menu's Copy Last item (a shortcut like Vella's Copy
/// Last Transcript, "Copy last image or output"). A window records its result when the run delivers it; the menu copies
/// it as the window's own Copy would: a file as PNG/TIFF or the file, text as text. Kept across launches; a result
/// whose file is gone offers nothing.
@MainActor final class LastOutput {
    enum Content: Equatable, Codable {
        case file(URL)
        case text(String)
    }
    struct Item: Equatable, Codable {
        var content: Content
        /// "Image", "Transcript", "Sound": the menu reads "Copy Last <noun>".
        var noun: String
        var model: String
        var made: Date
    }

    static let defaultsKey = "LastOutput"
    static let recentKey = "RecentOutputs"
    /// How many results the menu's Recent lists.
    static let recentCount = 5
    private let defaults: UserDefaults
    /// Where a text result is kept across launches (a long transcript does not belong in preferences); without one,
    /// text lasts until quit.
    private let textFile: URL?
    private(set) var item: Item?
    /// The newest results, newest first (files only kept while their file exists; text only the newest, as before).
    private(set) var history: [Item] = []
    /// The menu rebuilds when the item changes.
    var changed: () -> Void = {}

    init(defaults: UserDefaults = .standard, textFile: URL? = nil) {
        self.defaults = defaults
        self.textFile = textFile
        history =
            defaults.data(forKey: Self.recentKey).flatMap { try? JSONDecoder().decode([Item].self, from: $0) }?.filter {
                if case .file = $0.content { return true }
                return false
            } ?? []
        guard var saved = defaults.data(forKey: Self.defaultsKey).flatMap({ try? JSONDecoder().decode(Item.self, from: $0) }) else { return }
        if case .text = saved.content {
            guard let textFile, let text = try? String(contentsOf: textFile, encoding: .utf8) else { return }
            saved.content = .text(text)
        }
        item = saved
        // The newest result heads Recent; a file kept by a build that had only Copy Last joins it too.
        if case .text = saved.content { history.insert(saved, at: 0) } else if !history.contains(saved) { history.insert(saved, at: 0) }
    }

    func record(_ content: Content, noun: String, model: String) {
        if case .text(let s) = content, s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return }
        let next = Item(content: content, noun: noun, model: model, made: Date())
        item = next
        var stored = next
        if case .text(let s) = content {
            stored.content = .text("")
            if let textFile {
                try? FileManager.default.createDirectory(at: textFile.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? s.write(to: textFile, atomically: true, encoding: .utf8)
            }
        }
        defaults.set(try? JSONEncoder().encode(stored), forKey: Self.defaultsKey)
        // Recent: one text (the newest) and the newest files; the same file once.
        history.removeAll { old in
            if case .text = content, case .text = old.content { return true }
            return old.content == content
        }
        history.insert(next, at: 0)
        history = Array(history.prefix(Self.recentCount))
        defaults.set(
            try? JSONEncoder().encode(
                history.filter {
                    if case .file = $0.content { return true }; return false
                }), forKey: Self.recentKey)
        changed()
    }
    /// A file result, named by its type.
    func record(file url: URL, mime: String? = nil, model: String) {
        record(.file(url), noun: Self.noun(file: url, mime: mime), model: model)
    }

    /// What the menu offers now: nil when nothing was made or its file has been moved or deleted.
    var available: Item? {
        guard let item else { return nil }
        if case .file(let url) = item.content, !FileManager.default.fileExists(atPath: url.path) { return nil }
        return item
    }
    var title: String? { available.map { "Copy Last \($0.noun)" } }
    var help: String? {
        guard let item = available else { return nil }
        switch item.content {
        case .file(let url): return "\(url.lastPathComponent)\nMade by \(item.model)"
        case .text(let s): return String(s.prefix(160)) + (s.count > 160 ? "\u{2026}" : "") + "\nMade by \(item.model)"
        }
    }

    @discardableResult func copy(to pasteboard: NSPasteboard = .general) -> Bool {
        guard let item = available else { return false }
        switch item.content {
        case .file(let url): return (try? ToolFiles.copy(url, to: pasteboard)) != nil
        case .text(let s):
            pasteboard.clearContents()
            return pasteboard.setString(s, forType: .string)
        }
    }

    // MARK: Recent

    /// The results the menu's Recent offers now: newest first, without files that were moved or deleted.
    var recent: [Item] {
        history.filter { entry in
            if case .file(let url) = entry.content { return FileManager.default.fileExists(atPath: url.path) }
            return true
        }
    }
    /// "Sound · Kokoro 82M", the row's title; its subtitle is the file's name or the text's start.
    static func title(_ entry: Item) -> String { "\(entry.noun) \u{00b7} \(entry.model)" }
    static func subtitle(_ entry: Item) -> String {
        switch entry.content {
        case .file(let url): return url.lastPathComponent
        case .text(let s):
            let line = s.split(separator: "\n").first.map(String.init) ?? s
            return line.count > 48 ? String(line.prefix(47)) + "\u{2026}" : line
        }
    }
    static func symbol(_ entry: Item) -> String {
        switch entry.noun {
        case "Image": return "photo"
        case "Sound": return "waveform"
        case "Video": return "film"
        case "3D Model": return "cube"
        default: return "doc.text"
        }
    }
    private func entry(_ index: Int) -> Item? { recent.indices.contains(index) ? recent[index] : nil }
    func open(_ index: Int) {
        guard case .file(let url)? = entry(index)?.content else { return }
        NSWorkspace.shared.open(url)
    }
    func reveal(_ index: Int) {
        guard case .file(let url)? = entry(index)?.content else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
    @discardableResult func copy(_ index: Int, to pasteboard: NSPasteboard = .general) -> Bool {
        guard let entry = entry(index) else { return false }
        switch entry.content {
        case .file(let url): return (try? ToolFiles.copy(url, to: pasteboard)) != nil
        case .text(let s):
            pasteboard.clearContents()
            return pasteboard.setString(s, forType: .string)
        }
    }
    /// Save As…: a copy where the person chooses (a text result as a .txt file).
    func saveAs(_ index: Int) {
        guard let entry = entry(index) else { return }
        let panel = NSSavePanel()
        switch entry.content {
        case .file(let url):
            panel.nameFieldStringValue = url.lastPathComponent
            panel.directoryURL = url.deletingLastPathComponent()
        case .text: panel.nameFieldStringValue = "\(entry.noun).txt"
        }
        NSApp.activate(ignoringOtherApps: true)
        guard panel.runModal() == .OK, let target = panel.url else { return }
        Self.write(entry.content, to: target)
    }
    /// Writes a result to a chosen place, replacing what the person agreed to replace there.
    @discardableResult static func write(_ content: Content, to target: URL) -> Bool {
        let fm = FileManager.default
        switch content {
        case .file(let url):
            if url.standardizedFileURL == target.standardizedFileURL { return true }
            try? fm.removeItem(at: target)
            return (try? fm.copyItem(at: url, to: target)) != nil
        case .text(let s): return (try? s.write(to: target, atomically: true, encoding: .utf8)) != nil
        }
    }

    static func noun(file url: URL, mime: String?) -> String {
        let type = mime ?? ""
        let ext = url.pathExtension.lowercased()
        if type.hasPrefix("image/") || ["png", "jpg", "jpeg", "webp", "heic", "tif", "tiff"].contains(ext) { return "Image" }
        if type.hasPrefix("audio/") || ["wav", "mp3", "m4a", "flac", "aiff", "caf", "ogg"].contains(ext) { return "Sound" }
        if type.hasPrefix("video/") || ["mp4", "mov", "m4v"].contains(ext) { return "Video" }
        if type.hasPrefix("model/") || ["glb", "gltf", "obj", "usdz", "ply"].contains(ext) { return "3D Model" }
        return "Output"
    }
}
