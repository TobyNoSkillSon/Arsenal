// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// The model browser's data and logic as plain values: one item per catalog entry, the sidebar's
// filter state (kind, modality, Fits, facet values, search), the sort, the count line and each row's expanded details.
// No AppKit: unit-tested directly; BrowserWindow / BrowserList / BrowserSidebar / ModelDetailView draw it.

/// One row of the browser, from the service's `CatalogEntry` and the kind it belongs to.
struct BrowserItem: Equatable, Identifiable {
    /// The download behind a 16-bit copy that is smaller on disk: an FP32 source cast to FP16 as it arrives (RIFE
    /// downloads 23 MB and keeps 11 MB). Nil for every other copy, whose download is its size.
    static func castDownload(_ e: CatalogEntry, _ d: String) -> Int64? {
        guard e.sourceDtype == "fp32", d != "fp32", let size = e.sizesByDtype[d], size > 0, let fetch = e.downloadBytes[d], fetch > size else { return nil }
        return fetch
    }
    /// The Status column: only a state that changes what you do with the model; blank otherwise. How far the model was verified is in the name's hover.
    enum Status: String, CaseIterable { case leaving = "Leaving catalog", failedChecks = "Checks failed", update = "Update", local = "Local", none = "" }
    /// How far the model was verified, for the name's hover.
    enum Grade: String { case measured = "Measured", quickCheck = "Quick check", unverified = "Unverified", local = "Local" }
    var entry: CatalogEntry
    var id: String
    var name: String
    /// What the model does ("Image generation"), under its name.
    var purpose: String
    var kind: String
    var kindTitle: String
    var kindOrder: Int
    var icon: String
    /// The kind's figure columns (Peak RAM included), for a runner's recorded numbers.
    var columns: [KindColumn]
    var suggested: Bool
    var downloads: Double
    /// The generic tool's menu title for this kind ("Separate Layers"): its primary operation's title.
    var primaryOperationTitle: String?
    var released: Int
    var params: String
    var paramsValue: Double
    /// Bytes still to fetch for the default dtype (0: already on this Mac).
    var downloadBytes: Int64
    /// Bytes of the default dtype a copy already on this Mac covers (another app's folder, the HF cache).
    var localCopyBytes: Int64
    /// The default dtype's weights are already in Arsenal's store (the table's Ready/Loaded).
    var onThisMac: Bool
    /// Every offered dtype's tooltip part: "bf16 55.6 GB", "int8 made here", "int4 1.9 GB (vendor)".
    var downloadParts: [String]
    var ramGB: Double
    /// The shown dtype's peak memory from the Models table (Figures.peak): one source for both windows; nil draws the
    /// table's yellow triangle, never the service's estimate.
    var peak: String?
    var peakHelp: String = ""
    /// The precision the model is loaded at (the table's blue row, flame and white loaded edge), nil when not loaded.
    var loadedDtype: String?
    /// The precision chosen in the Models table (its gold cell), nil for a model not in the table.
    var chosenDtype: String?
    /// int4's quality warning (the table's orange triangle on int4), nil when int4 passed its check or is not offered.
    var int4Warning: String?
    /// Custom kernels for this chip: the table's bolt before the formats.
    var fastKernels = false
    /// The service's RAM figure (MB) behind `ramGB`: Fits compares it, as the service does (`CatalogMatch.fits`).
    var ramMB: Double
    /// Offered dtypes, widest first (any format label a folder declares, v2.2 item 5).
    var dtypes: [String]
    /// What the row shows at each offered dtype; a dtype filter projects the row onto the dtype that matched.
    struct Figures: Equatable {
        /// Weights at this dtype (an estimate for one made here or by the runner).
        var size: Int64?
        var estimate: Bool
        var download: Int64, localCopy: Int64
        var onThisMac: Bool
        var ramMB: Double
        var ramGB: Double { ramMB / 1000 }
        /// The Models table's peak memory at this dtype, when one is known (measured or derived): shown instead of the
        /// estimate, so both windows print the same figure. Fits and sorting keep the service's estimate.
        var peak: String?
        /// The table's hover for that figure, or why it is missing (the cell then shows the table's yellow triangle).
        var peakHelp: String = ""
    }
    var figures: [String: Figures]
    var defaultDtype: String
    /// The dtype a dtype/Size/RAM/Fits filter matched (the row shows its numbers), nil when unfiltered.
    var matchedDtype: String?
    var licence: String
    var licenceURL: String
    var licenceName: String?
    var licenceConditions: String?
    var commercial: Bool
    var languages: [String]
    var status: Status
    var statusHelp: String
    var added: Bool
    var weightsPresent: Bool
    var modalitiesIn: [String]
    var modalitiesOut: [String]
    /// Lowercased text the search matches: name, id, publisher, kind, tasks, languages (codes and names), licence, repo.
    var searchText: String
    /// The Model cell's tooltip (publisher, year, licence, summary, params, runner).
    var help: String

    /// `shown`: the group it shows in (KindPresentation.id with the service's kinds); absent, from its kind and tasks.
    init(_ e: CatalogEntry, kind k: KindSpec?, shown: String? = nil) {
        entry = e
        id = e.id; name = e.name; purpose = e.row.purpose
        // Shown by what it does (KindPresentation): a chat-kind model that reads documents is in Documents.
        kind = shown ?? KindPresentation.id(kind: e.kind, tasks: e.tasks)
        // The group's own name, as the sidebar and the Models table name it (one word per concept).
        kindTitle = k?.title ?? e.kind.capitalized
        kindOrder = k?.order ?? Int.max; icon = k?.icon ?? KindIcon.fallback(kind)
        columns = k.map(MetricText.columns) ?? [MetricText.peakRAM]
        suggested = e.suggested; downloads = e.popularity["hf_downloads"]?.number ?? 0
        primaryOperationTitle = ToolCenter.primaryOperation(k)?.title
        released = e.released ?? 0
        params = e.params.isEmpty ? "" : ParamsText.display(e.params, name: e.name); paramsValue = Self.paramsValue(e.params)
        // 8-bit is not offered for any model: not listed, not the default, no figures. A saved 8-bit choice is
        // the 16-bit copy.
        let offered = Self.ordered(e).filter { !PrecisionColumn.isEightBit($0, formats: e.formats) }
        dtypes = offered
        let saved = PrecisionColumn.offered(e.row.selectedPrecision, e.row)
        let defaultDtype = offered.contains(saved) ? saved : offered.first ?? saved
        self.defaultDtype = defaultDtype
        let have: Set<ModelState> = [.ready, .loaded, .loading]
        let derived = Set(e.derivedDtypes)
        var figures: [String: Figures] = [:]
        for d in Set(offered + [defaultDtype]) {
            let present = have.contains(e.row.states[d]?.state ?? .notInstalled)
            let format = e.formats.first { $0.id == d }
            let made = derived.contains(d) || format?.source == .derived || format?.source == .runnerNative
            let memory = TableFigures.figure(e.row, precision: d, column: .memory, host: nil)
            figures[d] = Figures(
                size: e.sizesByDtype[d], estimate: made && format?.sizeBytes == nil,
                download: present ? 0 : e.downloadBytes[d] ?? e.sizesByDtype[d] ?? (d == defaultDtype ? e.row.sizeBytes : nil) ?? 0,
                localCopy: present ? 0 : e.localCopyBytes[d] ?? 0, onThisMac: present,
                ramMB: d == defaultDtype ? e.minRAMMB : e.ramMBByDtype[d] ?? e.minRAMMB,
                peak: memory.value, peakHelp: memory.help)
        }
        self.figures = figures
        let shown = figures[defaultDtype]!
        downloadBytes = shown.download; localCopyBytes = shown.localCopy; onThisMac = shown.onThisMac
        downloadParts = dtypes.map { d in
            if derived.contains(d) { return "\(d) made here" }
            let bytes = e.sizesByDtype[d] ?? e.downloadBytes[d] ?? 0
            // A download larger than its copy on disk (an FP32 source cast to FP16) names both.
            let fetch = Self.castDownload(e, d)
            let source = fetch.map { " from a \(formatBytes($0)) download" } ?? ""
            return "\(d) \(formatBytes(bytes))" + source + (PrecisionColumn.of(d, formats: e.formats) == "16" ? "" : " (vendor)")
        }
        ramGB = shown.ramGB; ramMB = shown.ramMB; peak = shown.peak; peakHelp = shown.peakHelp
        loadedDtype = e.row.residentDType; fastKernels = e.row.hasFastKernels
        chosenDtype = e.added ? defaultDtype : nil
        int4Warning = e.dtypes.contains("int4") ? e.row.precisionQualityWarnings["int4"] : nil
        licence = e.licence; licenceURL = e.licenceURL; commercial = e.commercial
        licenceName = e.licenceName; licenceConditions = e.licenceConditions
        languages = e.languages
        let grade: Grade =
            e.local ? .local : e.row.badge != .unverified ? .measured : e.row.verification == "quick check" ? .quickCheck : .unverified
        if e.row.catalogRemoved {
            status = .leaving
        } else if e.row.checks?.label == "Failed" {
            status = .failedChecks
        } else if e.row.precisions.contains(where: { e.row.states[$0]?.state == .updateAvailable }) {
            status = .update
        } else if e.local {
            status = .local
        } else {
            status = .none
        }
        statusHelp = Self.statusHelp(status, e.row)
        added = e.added; weightsPresent = e.installed
        // A folder without its own [facets] has its kind's (derivable data is derived, CONTRACTS §15).
        modalitiesIn = e.modalitiesIn.isEmpty ? k?.facets?.modalitiesIn ?? [] : e.modalitiesIn
        modalitiesOut = e.modalitiesOut.isEmpty ? k?.facets?.modalitiesOut ?? [] : e.modalitiesOut
        let languageNames = e.languages.map(Languages.name)
        searchText = ([e.name, e.id, e.row.purpose, e.publisher, kindTitle, k?.title ?? "", e.licence, e.licenceName ?? "", e.hfRepo] + e.tasks + e.languages + languageNames)
            .joined(separator: " ").lowercased()
        var lines = [e.name]
        let origin = [e.publisher.isEmpty ? nil : e.publisher, e.row.released.map(String.init)].compactMap { $0 }.joined(separator: ", ")
        let first = [
            origin.isEmpty ? nil : origin,
            e.licence.isEmpty ? nil : ModelHelp.licenceLine(LicenceName.full(e.licence, name: e.licenceName), commercial: e.commercial, conditions: .some(e.licenceConditions))
        ].compactMap { $0 }.joined(separator: " \u{00b7} ")
        if !first.isEmpty { lines.append(first) }
        if !e.summary.isEmpty { lines.append(e.summary) }
        if !e.params.isEmpty { lines.append("\(params) parameters") }
        lines.append(ModelHelp.runnerLine(e.row.runner, language: e.row.runnerLanguage))
        if grade != .local { lines.append(grade.rawValue + ": " + Self.gradeHelp(grade).prefix(1).lowercased() + Self.gradeHelp(grade).dropFirst()) }
        help = lines.joined(separator: "\n")
    }

    /// Widest first by bits per weight, the service's order (`CatalogMatch`), so a filter matches the same dtype here
    /// as in `arsenal models`.
    static func ordered(_ e: CatalogEntry) -> [String] { CatalogMatch.orderedDtypes(e) }
    /// This row as it reads at dtype `d`: Download, RAM and the dtype cell follow it.
    func projected(_ d: String) -> BrowserItem {
        guard let f = figures[d] else { return self }
        var item = self
        item.matchedDtype = d; item.downloadBytes = f.download; item.localCopyBytes = f.localCopy; item.onThisMac = f.onThisMac
        item.ramGB = f.ramGB; item.ramMB = f.ramMB; item.peak = f.peak; item.peakHelp = f.peakHelp
        return item
    }

    static func paramsValue(_ text: String) -> Double {
        let t = text.uppercased().trimmingCharacters(in: .whitespaces)
        let scale: Double = t.hasSuffix("B") ? 1e9 : t.hasSuffix("M") ? 1e6 : t.hasSuffix("K") ? 1e3 : 1
        return (Double(t.trimmingCharacters(in: .letters)) ?? 0) * scale
    }
    static func statusHelp(_ s: Status, _ row: ModelRow) -> String {
        switch s {
        case .leaving: return "No longer in the catalog: remove it from the table; you choose whether to keep the weights."
        case .failedChecks: return row.checks.map(ModelHelp.checks) ?? "Its checks failed."
        case .update: return "A new revision is available: Update in the Models table."
        case .local: return ModelHelp.local(row)
        case .none: return ""
        }
    }
    static func gradeHelp(_ g: Grade) -> String {
        switch g {
        case .measured: return "Passed its quality gate, with measured figures."
        case .quickCheck: return "A quick check ran (uncontrolled); not ranked."
        case .unverified: return "Not tried on this Mac yet."
        case .local: return "A folder in your local models folder: not reviewed or signed by Arsenal."
        }
    }

    // MARK: Cell text

    /// What the table's Get says: a size, `On this Mac` (Get adds the copy already here) or, when the weights are in
    /// the store, nothing written: the cell draws a quiet checkmark (`downloadCheck`).
    var updateAvailable: Bool { entry.row.states[matchedDtype ?? defaultDtype]?.state == .updateAvailable }
    var downloadText: String {
        updateAvailable
            ? (downloadBytes > 0 ? "Update · \(formatBytes(downloadBytes))" : "Update")
            : downloadBytes > 0 ? formatBytes(downloadBytes) : localCopyBytes > 0 ? "On this Mac" : downloadCheck ? "" : "\u{2014}"
    }
    var downloadCheck: Bool { downloadBytes == 0 && localCopyBytes == 0 && onThisMac }
    var downloadHelp: String {
        if updateAvailable {
            return "Update in the Models table before loading this precision." + (downloadBytes > 0 ? " Downloads \(formatBytes(downloadBytes))." : "")
        }
        guard downloadBytes == 0 else {
            let parts = downloadParts.joined(separator: " \u{00b7} ")
            return matchedDtype.map { "Get at \($0) downloads \(formatBytes(downloadBytes))\n" + parts } ?? parts
        }
        return localCopyBytes > 0 ? "On this Mac: Get checks and adds the copy already here (\(formatBytes(localCopyBytes))); nothing to download" : "On this Mac"
    }
    /// The Models table's peak memory figure; empty when it has none (the cell draws the table's triangle).
    var ramText: String { peak ?? "" }
    /// Red Peak memory cell: the service's Fits says no (its units and headroom); the help shows memory as the Mac names it.
    func ramExceeds(physicalBytes: UInt64) -> Bool { !CatalogMatch.fits(ramMB: ramMB, physicalBytes: physicalBytes) }
    /// The table's hover for the figure (or why it is missing), then whether it fits this Mac.
    func ramHelp(physicalBytes: UInt64) -> String? {
        let gb = (Double(physicalBytes) / 1_073_741_824).rounded()
        let line = (gb * MacMemoryGuard.safeInUseFraction).rounded(.down)
        let need = MemoryDisplay.gib(decimalGB: ramGB).rounded(.up)
        let fits =
            ramExceeds(physicalBytes: physicalBytes)
            ? String(format: "Needs about %.0f GB; Arsenal keeps this %.0f GB Mac under %.0f GB.", need, gb, line) : nil
        let lines = [peakHelp.isEmpty ? nil : peakHelp, fits].compactMap { $0 }
        return lines.isEmpty ? nil : lines.joined(separator: "\n")
    }
    var dtypeText: String { dtypes.joined(separator: " \u{00b7} ") }
    /// The licence as people name it ("Apache 2.0", "CC BY-NC 4.0", "SAM License"), never a raw SPDX or LicenseRef id.
    var licenceText: String { LicenceName.short(licence, name: licenceName) }
    var licenceUse: String { commercial ? "commercial use allowed" : licenceConditions ?? "non-commercial" }
    /// The hover: why the name is orange first (commercial use is limited), then its full name and where it is.
    var licenceHelp: String {
        let full = LicenceName.full(licence, name: licenceName)
        let lines =
            commercial
            ? [full, "Commercial use allowed.", licenceURL]
            : ["Orange: commercial use is limited.", full, licenceConditions ?? "Non-commercial use only.", licenceURL]
        return lines.filter { !$0.isEmpty }.joined(separator: "\n")
    }
    /// In words, never codes: one language by name, several as a count (the hover names them all).
    var languagesText: String {
        if languages.contains("multi") || languages.contains("mul") { return Languages.name("multi") }
        if languages == ["und"] { return Languages.name("und") }
        if languages.count == 1 { return Languages.name(languages[0]) }
        return languages.isEmpty ? "" : "\(languages.count) languages"
    }
    var languagesHelp: String { languages.map(Languages.name).joined(separator: ", ") }
    /// VoiceOver: "<name>, <kind>, <params>, <download>, <status>".
    var accessibilityLabel: String {
        [name, kindTitle, params, downloadCheck ? "on this Mac" : downloadText, status.rawValue]
            .filter { !$0.isEmpty }.joined(separator: ", ")
    }
}

/// Licences by the names people know them by: short in a cell or a pill, full in a hover.
enum LicenceName {
    static let known: [String: String] = [
        "apache-2.0": "Apache 2.0", "mit": "MIT", "bsd-2-clause": "BSD 2-Clause", "bsd-3-clause": "BSD 3-Clause", "cc-by-4.0": "CC BY 4.0",
        "cc-by-sa-4.0": "CC BY-SA 4.0", "cc-by-nc-4.0": "CC BY-NC 4.0", "cc-by-nc-sa-4.0": "CC BY-NC-SA 4.0", "cc0-1.0": "CC0", "gpl-3.0": "GPL 3.0",
        "openmdw-1.1": "OpenMDW 1.1", "sam-license": "SAM License", "licenseref-sam": "SAM License", "nvidia-research": "NVIDIA Research",
        "licenseref-openrail": "OpenRAIL-M", "licenseref-pending": "Pending", "licenseref-lfm-open-1.0": "LFM Open 1.0",
        "licenseref-nvidia-nsclv1": "NVIDIA NSCL v1", "licenseref-stability-ai-community": "Stability AI Community",
        "licenseref-breezeblue-rnc-1.1": "BreezeBlue RNC 1.1"
    ]
    /// The short name: a known licence's, else a folder's own name when it is a name (not an id), else the id in words.
    static func short(_ spdx: String, name: String?) -> String {
        if let k = known[spdx.lowercased()] { return k }
        if let name, !name.isEmpty, !name.hasPrefix("LicenseRef-"), name.count <= 24 { return name }
        let id = spdx.hasPrefix("LicenseRef-") ? String(spdx.dropFirst(11)) : spdx
        return id.replacingOccurrences(of: "-", with: " ")
    }
    /// The full name for the hover: the folder's own name when it is a name, else the short one.
    static func full(_ spdx: String, name: String?) -> String {
        guard let name, !name.isEmpty, !name.hasPrefix("LicenseRef-"), name != spdx else { return short(spdx, name: name) }
        return name
    }
}

enum Languages {
    static func name(_ code: String) -> String {
        if code == "multi" || code == "mul" { return "100+ languages" }
        if code == "und" { return "Any language" }
        return Locale(identifier: "en").localizedString(forLanguageCode: code) ?? code
    }
}

enum KindIcon {
    /// Kinds whose toml has no `icon` yet.
    static func fallback(_ kind: String) -> String {
        [
            "voices": "waveform", "chat": "text.bubble", "images": "photo", "decisions": "checkmark.circle", "sound": "music.note",
            "enhancement": "wand.and.stars"
        ][kind] ?? "cube"
    }
}

/// The sidebar's facet sections (spec §1.3), in sidebar order. Values are pills: OR inside a facet, AND across.
enum BrowserFacet: String, CaseIterable, Identifiable {
    case params = "Params", licence = "Licence", dtype = "dtype", size = "Size", ram = "RAM", language = "Language", status = "Status", library = "Library"
    var id: String { rawValue }
    /// The section's name, as the Models table names the same things.
    var name: String { self == .dtype ? "Precision" : self == .ram ? "Peak memory" : rawValue }
    static let commercialValue = "Commercial use", nonCommercialValue = "Non-commercial"
    static let inTable = "In the table", notAdded = "Not added", onThisMac = "On this Mac"
    static let paramRanges = ["< 1B", "1\u{2013}4B", "4\u{2013}12B", "12\u{2013}32B", "> 32B"]
    static let sizeRanges = ["< 1 GB", "1\u{2013}8 GB", "8\u{2013}32 GB", "> 32 GB"]
    static let ramRanges = ["< 8 GB", "8\u{2013}16 GB", "16\u{2013}32 GB", "32\u{2013}64 GB", "> 64 GB"]
    /// dtype, Size and RAM hold for one offered dtype together (with Fits): "int4 · 1–8 GB" matches a model whose int4
    /// is 1–8 GB, whatever its default dtype weighs (v2.2 item 4).
    var perDtype: Bool { self == .dtype || self == .size || self == .ram }
    /// The section header's tooltip.
    var help: String? {
        switch self {
        case .dtype: return "Formats the model is offered in. With Size or RAM, all must hold for one format."
        case .size: return "The weights at a chosen format (or any format), not the default one."
        case .ram: return "Memory needed at a chosen format (or any format)."
        default: return nil
        }
    }
    /// Open by default (the rest start folded).
    var openByDefault: Bool { self == .params || self == .licence }

    /// The values offered, from what the catalog contains where that is data.
    func values(_ items: [BrowserItem]) -> [String] {
        switch self {
        case .params: return Self.paramRanges
        case .size: return Self.sizeRanges
        case .ram: return Self.ramRanges
        case .dtype:
            // Every format in the catalog, widest first.
            var bits: [String: Double] = [:]
            for item in items { for d in item.dtypes { bits[d] = PrecisionColumn.bits(d, formats: item.entry.formats) ?? 0 } }
            return bits.keys.sorted { bits[$0]! != bits[$1]! ? bits[$0]! > bits[$1]! : $0 < $1 }
        case .licence:
            let names = Array(Set(items.filter(\.commercial).map(\.licenceText)).filter { !$0.isEmpty }).sorted()
            return [Self.commercialValue] + names + (items.contains { !$0.commercial } ? [Self.nonCommercialValue] : [])
        case .language:
            var counts: [String: Int] = [:]
            for code in items.flatMap(\.languages) { counts[code, default: 0] += 1 }
            return counts.keys.sorted { counts[$0]! != counts[$1]! ? counts[$0]! > counts[$1]! : Languages.name($0) < Languages.name($1) }
        case .status: return BrowserItem.Status.allCases.filter { $0 != .none }.map(\.rawValue).filter { v in items.contains { $0.status.rawValue == v } }
        case .library: return [Self.inTable, Self.notAdded, Self.onThisMac]
        }
    }
    /// How a value reads on its pill (language codes by name).
    func title(_ value: String) -> String { self == .language ? Languages.name(value) : value }

    /// `dtype`: the offered dtype a per-dtype facet (dtype, Size, RAM) is checked at.
    func matches(_ item: BrowserItem, _ chosen: Set<String>, dtype: String? = nil) -> Bool {
        guard !chosen.isEmpty else { return true }
        switch self {
        case .params: return chosen.contains(Self.paramRange(item.paramsValue))
        case .dtype: return dtype.map { chosen.contains($0) } ?? !chosen.isDisjoint(with: item.dtypes)
        case .size:
            guard let d = dtype ?? Optional(item.defaultDtype), let size = item.figures[d]?.size else { return false }
            return chosen.contains(Self.sizeRange(size))
        case .ram:
            guard let d = dtype ?? Optional(item.defaultDtype), let ram = item.figures[d]?.ramGB else { return false }
            return chosen.contains(Self.ramRange(ram))
        case .licence:
            return (chosen.contains(Self.commercialValue) && item.commercial) || (chosen.contains(Self.nonCommercialValue) && !item.commercial)
                || chosen.contains(item.licenceText)
        case .language: return !chosen.isDisjoint(with: item.languages)
        case .status: return chosen.contains(item.status.rawValue)
        case .library:
            return (chosen.contains(Self.inTable) && item.added) || (chosen.contains(Self.notAdded) && !item.added) || (chosen.contains(Self.onThisMac) && item.weightsPresent)
        }
    }
    static func paramRange(_ n: Double) -> String {
        n < 1e9 ? paramRanges[0] : n < 4e9 ? paramRanges[1] : n < 12e9 ? paramRanges[2] : n <= 32e9 ? paramRanges[3] : paramRanges[4]
    }
    static func sizeRange(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1e9
        return gb < 1 ? sizeRanges[0] : gb < 8 ? sizeRanges[1] : gb <= 32 ? sizeRanges[2] : sizeRanges[3]
    }
    static func ramRange(_ gb: Double) -> String {
        gb < 8 ? ramRanges[0] : gb < 16 ? ramRanges[1] : gb < 32 ? ramRanges[2] : gb <= 64 ? ramRanges[3] : ramRanges[4]
    }
}

/// The modality grid's columns (spec §1.3 item 7).
enum Modality: String, CaseIterable, Identifiable {
    case text, audio, image, video
    var id: String { rawValue }
    var title: String { rawValue.capitalized }
    var symbol: String { ["text": "textformat", "audio": "waveform", "image": "photo", "video": "film"][rawValue]! }
}

/// The list's columns, left to right.
enum BrowserColumn: String, CaseIterable {
    case chevron, icon, model = "Model", kind = "Kind", params = "Params", download = "Download", ram = "RAM", dtype = "dtype"
    case licence = "Licence", languages = "Languages", status = "Status", flex, action
    /// The heading, as the Models table names the same figures (the raw value stays the saved column identifier).
    var title: String { self == .ram ? "Peak memory" : self == .dtype ? "Precision" : rawValue }
}

/// The sort keys (sidebar pop-up and header clicks drive the same state).
enum SortKey: String, CaseIterable {
    case downloads = "Most downloaded", newest = "Newest", name = "Name", params = "Params", download = "Download size"
    case ram = "RAM", licence = "Licence", status = "Status", kind = "Kind", dtype = "dtype", languages = "Languages"
    /// The name shown, as the Models table names the same figures.
    var title: String { self == .ram ? "Peak memory" : self == .dtype ? "Precision" : rawValue }
    /// Keys the sidebar's pop-up lists (the header can sort by the rest too).
    static let menu: [SortKey] = [.downloads, .newest, .name, .params, .download, .ram, .licence, .status]
    /// The header column a key belongs to.
    var column: BrowserColumn? {
        switch self {
        case .downloads, .newest: return nil
        case .name: return .model
        case .params: return .params
        case .download: return .download
        case .ram: return .ram
        case .licence: return .licence
        case .status: return .status
        case .kind: return .kind
        case .dtype: return .dtype
        case .languages: return .languages
        }
    }
    static func of(_ column: BrowserColumn) -> SortKey? { allCases.first { $0.column == column } }
}

struct BrowserSort: Equatable {
    /// Arsenal recommends no model: the list opens most downloaded first.
    var key: SortKey = .downloads
    /// The key's natural order (Most downloaded and Newest put the most first); false reverses it.
    var ascending = true
    static let `default` = BrowserSort()
    /// Finder: another column sorts in its natural order; the same column again reverses.
    mutating func click(_ c: BrowserColumn) {
        guard let k = SortKey.of(c) else { return }
        if k == key { ascending.toggle() } else { key = k; ascending = true }
    }
}

/// What the browser shows: the sidebar's filters, the search and the sort.
struct BrowserState: Equatable {
    var kind: String?
    var fits = false
    var inputs: Set<String> = [], outputs: Set<String> = []
    var values: [BrowserFacet: Set<String>] = [:]
    var search = ""
    var sort = BrowserSort.default
    var filtering: Bool {
        kind != nil || fits || !inputs.isEmpty || !outputs.isEmpty || values.values.contains { !$0.isEmpty } || !search.trimmingCharacters(in: .whitespaces).isEmpty
    }

    mutating func toggle(_ f: BrowserFacet, _ value: String) {
        var set = values[f] ?? []
        if set.contains(value) { set.remove(value) } else { set.insert(value) }
        values[f] = set.isEmpty ? nil : set
    }
    mutating func toggle(input m: Modality) { if inputs.contains(m.rawValue) { inputs.remove(m.rawValue) } else { inputs.insert(m.rawValue) } }
    mutating func toggle(output m: Modality) { if outputs.contains(m.rawValue) { outputs.remove(m.rawValue) } else { outputs.insert(m.rawValue) } }
    mutating func clearFacet(_ f: BrowserFacet) { values[f] = nil }
    /// `Clear`: search, kind, modality and every facet go; the sort stays.
    mutating func clearFilters() { kind = nil; fits = false; inputs = []; outputs = []; values = [:]; search = "" }
    /// A kind row: replaces the selection; the selected one again returns to All.
    mutating func choose(kind k: String?) { kind = (k == kind) ? nil : k }

    /// AND across kind, Fits, modality, facets and every search term; OR inside each. Fits, dtype, Size and RAM must
    /// all hold at one offered dtype, picked by the service's own rule (`CatalogMatch`: the default first, then widest
    /// first; Fits in its units with its headroom); the row is then shown at that dtype.
    func visible(_ items: [BrowserItem], physicalBytes: UInt64) -> [BrowserItem] {
        let terms = search.lowercased().split(whereSeparator: \.isWhitespace).map(String.init)
        let perDtype = values.filter { $0.key.perDtype && !$0.value.isEmpty }
        let shown = items.compactMap { item -> BrowserItem? in
            let general =
                (kind == nil || item.kind == kind)
                && (inputs.isEmpty || !inputs.isDisjoint(with: item.modalitiesIn)) && (outputs.isEmpty || !outputs.isDisjoint(with: item.modalitiesOut))
                && values.allSatisfy { $0.key.perDtype || $0.key.matches(item, $0.value) }
                && terms.allSatisfy { item.searchText.contains($0) }
            guard general else { return nil }
            guard fits || !perDtype.isEmpty else { return item }
            let match = CatalogMatch.matchingDtype(item.entry) { d in
                (!fits || item.figures[d].map { CatalogMatch.fits(ramMB: $0.ramMB, physicalBytes: physicalBytes) } == true)
                    && perDtype.allSatisfy { $0.key.matches(item, $0.value, dtype: d) }
            }
            return match.map { perDtype.isEmpty && $0 == item.defaultDtype ? item : item.projected($0) }
        }
        return Self.sorted(shown, sort)
    }

    static func sorted(_ items: [BrowserItem], _ sort: BrowserSort) -> [BrowserItem] {
        let statusRank: [BrowserItem.Status: Int] = [.leaving: 0, .failedChecks: 1, .update: 2, .local: 3, .none: 4]
        func compare(_ a: BrowserItem, _ b: BrowserItem) -> ComparisonResult {
            func num(_ x: Double, _ y: Double) -> ComparisonResult { x == y ? .orderedSame : x < y ? .orderedAscending : .orderedDescending }
            func text(_ x: String, _ y: String) -> ComparisonResult { x.localizedStandardCompare(y) }
            func then(_ r: ComparisonResult, _ next: @autoclosure () -> ComparisonResult) -> ComparisonResult { r == .orderedSame ? next() : r }
            switch sort.key {
            case .downloads: return num(b.downloads, a.downloads)
            case .newest: return num(Double(b.released), Double(a.released))
            case .name: return text(a.name, b.name)
            case .kind: return num(Double(a.kindOrder), Double(b.kindOrder))
            case .params: return num(a.paramsValue, b.paramsValue)
            case .download: return num(Double(a.downloadBytes), Double(b.downloadBytes))
            case .ram: return num(a.ramGB, b.ramGB)
            case .dtype: return then(num(Double(a.dtypes.count), Double(b.dtypes.count)), text(a.dtypeText, b.dtypeText))
            case .licence: return then(num(a.commercial ? 0 : 1, b.commercial ? 0 : 1), text(a.licenceText, b.licenceText))
            case .languages: return then(num(Double(a.languages.count), Double(b.languages.count)), text(a.languages.first ?? "", b.languages.first ?? ""))
            case .status: return num(Double(statusRank[a.status] ?? 9), Double(statusRank[b.status] ?? 9))
            }
        }
        return items.sorted { a, b in
            let r = compare(a, b)
            // Ties break by name, always A→Z (stable whichever way the key runs).
            if r == .orderedSame { return a.name.localizedStandardCompare(b.name) == .orderedAscending }
            return sort.ascending ? r == .orderedAscending : r == .orderedDescending
        }
    }

    /// `15 models · 2 added`, filtered `2 of 15 models · 2 added`.
    static func count(shown: Int, total: Int, added: Int, filtered: Bool) -> String {
        let models = filtered ? "\(shown) of \(total) models" : "\(total) \(total == 1 ? "model" : "models")"
        return "\(models) \u{00b7} \(added) added"
    }
    /// The collapsed toolbar's one line: `All kinds · Most downloaded`, or the kind and the active filters by name.
    func summary(kindTitle: String?) -> String {
        var parts = [kindTitle ?? "All kinds"]
        if fits { parts.append("Fits") }
        parts += Modality.allCases.filter { inputs.contains($0.rawValue) }.map { "\($0.title) in" }
        parts += Modality.allCases.filter { outputs.contains($0.rawValue) }.map { "\($0.title) out" }
        for f in BrowserFacet.allCases where !(values[f] ?? []).isEmpty {
            let chosen = values[f]!
            parts.append(chosen.count == 1 ? f.title(chosen.first!) : f.rawValue)
        }
        if !filtering || parts.count == 1 { parts.append(sort.key.rawValue) }
        return parts.joined(separator: " \u{00b7} ")
    }
    /// The empty list's line: `No models match`, with a kind `No voices match`.
    func noResults(kindTitle: String?) -> String { kindTitle.map { "No \($0.lowercased()) match" } ?? "No models match" }
}

// MARK: - Expanded details (spec §4.3, §4.4)

struct ModelDetail: Equatable {
    enum Tone: Equatable { case primary, secondary, amber }
    struct Line: Equatable { var key: String; var value: String; var tone: Tone = .primary; var note: String? }
    struct Link: Equatable { var title: String; var url: String }
    /// `label`: the precision as a person reads it ("16-bit", "4-bit").
    struct Size: Equatable { var dtype: String; var size: String; var note: String; var label: String = "" }
    struct RunnerFeature: Equatable, Identifiable {
        var id: String
        var symbol: String
        var help: String
        /// The feature in a few words, as the card says it.
        var title: String = ""
    }
    struct RunnerStat: Equatable, Identifiable {
        var title: String
        var value: String = ""
        var help: String = ""
        /// No figure: the yellow triangle, the reason in the hover.
        var missing = true
        /// The chip the figure was measured on when it is not this Mac's.
        var reference: String?
        var delta: TableFigure.Delta?
        var id: String { title }
    }
    struct Runner: Equatable, Identifiable {
        var name: String, engine: String, engineHelp: String
        var isDefault: Bool, selected: Bool, eligible: Bool
        var reason: String?
        var figures: [(String, String)]
        var figuresHelp: String
        var condition: String
        var stats: [RunnerStat] = []
        var features: [RunnerFeature] = []
        var precision: String = ""
        /// `precision` as the Models table and the Sizes line name it ("16-bit", "4-bit"): the chip's text.
        var precisionLabel: String = ""
        /// The model is loaded at `precision` with this runner: the gold format takes the table's white loaded edge.
        var precisionLoaded = false
        /// int4's quality warning when `precision` is a flagged int4: the table's orange triangle after it.
        var precisionWarning: String?
        /// The model is in the table, so `precision` is its chosen one (gold); otherwise the neutral cell.
        var precisionChosen = false
        var id: String { name }
        static func == (a: Runner, b: Runner) -> Bool {
            a.name == b.name && a.engine == b.engine && a.isDefault == b.isDefault && a.selected == b.selected && a.eligible == b.eligible && a.reason == b.reason
                && a.figures.map { $0.0 + $0.1 } == b.figures.map { $0.0 + $0.1 } && a.condition == b.condition
                && a.stats == b.stats && a.features == b.features && a.precision == b.precision && a.precisionLabel == b.precisionLabel
                && a.precisionLoaded == b.precisionLoaded && a.precisionWarning == b.precisionWarning && a.precisionChosen == b.precisionChosen
        }
    }
    var description: String
    var about: [Line]
    var links: [Link]
    var sizes: [Size]
    var runners: [Runner]
    /// What can be done from the details: open the model's tool (before it is loaded, too), switch back to the revision
    /// it ran before.
    enum Action: Equatable, Hashable {
        case openTool(String, title: String)
        case rollback
        /// The model's README, in a popover beside the action.
        case docs
        var title: String {
            switch self {
            case .openTool(_, let title): return title
            case .rollback: return "Use the previous revision"
            case .docs: return "Docs"
            }
        }
    }
    var actions: [Action] = []
    /// The runners can be chosen between (more than one eligible).
    var choosable: Bool
    /// The amber line under the runners (resident runner differs, or a Get is needed first).
    var note: String?

    init(_ item: BrowserItem, chip: String? = HostChip.current) {
        let e = item.entry, row = e.row
        description = e.description.isEmpty ? e.summary : e.description
        var about: [Line] = []
        if !e.publisher.isEmpty { about.append(Line(key: "Publisher", value: e.publisher)) }
        if let year = e.released { about.append(Line(key: "Released", value: String(year))) }
        if !e.licence.isEmpty { about.append(Line(key: "Licence", value: item.licenceText, tone: item.commercial ? .primary : .amber, note: item.licenceUse)) }
        if !e.languages.isEmpty { about.append(Line(key: "Languages", value: Self.languages(e.languages))) }
        let settings = e.settingsSummary.compactMap { s -> String? in
            let v = s.display(s.value)
            return v.isEmpty ? nil : "\(s.title) \(s.type == "toggle" ? v.lowercased() : v)"
        }
        if !settings.isEmpty { about.append(Line(key: "Settings", value: settings.joined(separator: " \u{00b7} "), tone: .secondary)) }
        // Checks take no line here (verification wording never takes a visible line in Get more models): a failure shows
        // as the row's Status, "Checks failed", whose hover has the breakdown (`ModelHelp.checks`).
        self.about = about
        // Docs leads, so later actions (a tool once the model is here) never push it sideways.
        actions.append(.docs)
        // Tools open from here before the model is loaded (a request loads it); voice has its controls in the menu.
        if row.revision?.previous != nil, row.revision?.swapping != true { actions.append(.rollback) }
        let installed = row.precisions.contains { [.ready, .loaded, .loading].contains(row.states[$0]?.state) }
        if e.added, installed {
            for tool in row.tools.compactMap(ToolKind.init(rawValue:)) where tool != .voice {
                let title = tool == .operation ? (item.primaryOperationTitle.map { $0 + "\u{2026}" } ?? tool.title) : tool.title
                actions.append(.openTool(tool.rawValue, title: title))
            }
        }
        let order = ["huggingface", "homepage", "paper", "licence"]
        let titles = ["huggingface": "Hugging Face", "homepage": "Homepage", "paper": "Paper", "licence": "Licence"]
        links = e.links.keys.sorted { (order.firstIndex(of: $0) ?? 99, $0) < (order.firstIndex(of: $1) ?? 99, $1) }
            .map { Link(title: titles[$0] ?? $0.prefix(1).uppercased() + $0.dropFirst(), url: e.links[$0]!) }
        sizes = Self.sizes(e)

        let selected = e.runners.first(where: \.selected)?.name ?? row.runner
        runners = e.runners.map { Self.runner($0, row: row, columns: item.columns, chip: chip) }
        if runners.isEmpty, !row.runner.isEmpty, row.runner != "Unavailable" {
            runners = [
                Runner(
                    name: row.runner, engine: Self.engine(row.runnerLanguage), engineHelp: "", isDefault: true, selected: true, eligible: true, reason: nil,
                    figures: [], figuresHelp: "", condition: Self.notMeasured(chip))
            ]
        }
        for index in runners.indices {
            let runnerName = runners[index].name
            var fallbackRunner = CatalogRunner(name: runnerName, language: row.runnerLanguage, engine: "")
            fallbackRunner.stats = row.precisionStats
            let catalogRunner =
                e.runners.first { $0.name == runnerName }
                ?? fallbackRunner
            runners[index].precision = row.selectedPrecision
            runners[index].precisionLabel = WeightsPresentation.bitLabel(row.selectedPrecision, formats: row.formats)
            runners[index].precisionLoaded = row.residentDType == row.selectedPrecision && (row.residentRunner ?? row.runner) == runnerName
            runners[index].precisionWarning = row.precisionQualityWarnings[row.selectedPrecision]
            runners[index].precisionChosen = item.added
            runners[index].stats = BrowserRunnerStats.rows(
                kind: e.kind, dtype: row.selectedPrecision, runner: catalogRunner, base: TableFigures.base(row), host: chip)
            runners[index].features = catalogRunner.features.map(BrowserRunnerStats.feature)
            // Current format support is authoritative; old passing measurements cannot restore removed support.
            if !e.formats.isEmpty, e.formats.first(where: { $0.id == row.selectedPrecision })?.runners.contains(runnerName) != true {
                runners[index].eligible = false
                runners[index].reason = row.selectedPrecision + ": This runner does not support this precision."
            } else if let set = catalogRunner.stats[row.selectedPrecision], set.availability.state == .unavailable {
                runners[index].eligible = false
                runners[index].reason = row.selectedPrecision + ": " + (set.availability.reason ?? "not available")
            }
        }
        choosable = runners.filter(\.eligible).count > 1
        let other = row.residentRunner.flatMap { $0 != selected ? $0 : nil }
        if row.states[row.selectedPrecision]?.state == .updateAvailable {
            let bytes = row.downloadBytes[row.selectedPrecision] ?? 0
            note = "Update in the Models table before loading" + (bytes > 0 ? " · \(formatBytes(bytes))" : "")
        } else if let other {
            note = "Loaded with \(other) until Reload"
        } else {
            note = nil
        }
    }

    static func languages(_ codes: [String]) -> String {
        let names = codes.map(Languages.name)
        return names.count > 8 ? names.prefix(8).joined(separator: ", ") + " +\(names.count - 8)" : names.joined(separator: ", ")
    }

    /// One line per offered dtype, widest first: the weights at that dtype (the service's figure; `~` where it is an
    /// estimate, for a dtype made here or by the runner) and where they come from. The first dtype made from a download
    /// that is not here yet names that download: `int8 ~1.2 GB from a 4.5 GB download`.
    static func sizes(_ e: CatalogEntry) -> [Size] {
        let row = e.row
        // 8-bit is not offered: its size is not listed either.
        let offered = BrowserItem.ordered(e).filter { !PrecisionColumn.isEightBit($0, formats: e.formats) }
        let have: Set<ModelState> = [.ready, .loaded]
        let anyHere = offered.contains { have.contains(row.states[$0]?.state ?? .notInstalled) }
        var toldSource = offered.contains { d in !made(e, d) && !have.contains(row.states[d]?.state ?? .notInstalled) }
        return offered.map { d in
            let format = e.formats.first { $0.id == d }
            let estimate = made(e, d) && format?.sizeBytes == nil
            let size = e.sizesByDtype[d].map { (estimate ? "~" : "") + formatBytes($0) } ?? "\u{2014}"
            let fetch = row.downloadBytes[d] ?? 0
            let note: String
            if have.contains(row.states[d]?.state ?? .notInstalled) {
                note = "on this Mac"
            } else if let reason = row.dtypeUnavailableReason(d) {
                note = reason.prefix(1).lowercased() + reason.dropFirst().trimmingCharacters(in: CharacterSet(charactersIn: "."))
            } else if !made(e, d) {
                // A download larger than the copy it leaves on disk (an FP32 source cast to FP16) says so.
                let cast = BrowserItem.castDownload(e, d)
                note =
                    (row.localCopyBytes[d] ?? 0) > 0
                    ? "on this Mac" : ModelHelp.removed(row, d) ? "deleted to save space, download again" : cast.map { "from a \(formatBytes($0)) download" } ?? "download"
            } else if format?.source == .runnerNative {
                note = "made by the runner as it loads"
            } else if anyHere, fetch > 0 {
                // Another dtype is here but its source is not any more: making this one downloads it again.
                note = "needs a \(formatBytes(fetch)) download"
            } else if !anyHere, !toldSource, fetch > 0 {
                note = "from a \(formatBytes(fetch)) download"; toldSource = true
            } else {
                note = "made here"
            }
            return Size(dtype: d, size: size, note: note, label: WeightsPresentation.bitLabel(d, formats: e.formats))
        }
    }
    /// Made on this Mac (a derive recipe) or by the runner, rather than downloaded as it is.
    static func made(_ e: CatalogEntry, _ d: String) -> Bool {
        let source = e.formats.first { $0.id == d }?.source
        return e.derivedDtypes.contains(d) || source == .derived || source == .runnerNative
    }

    static func engine(_ language: String) -> String {
        switch language.lowercased() {
        case "swift": return "Swift \u{00b7} MLX"
        case "": return ""
        default: return language.prefix(1).uppercased() + language.dropFirst()
        }
    }
    static func notMeasured(_ chip: String?) -> String { "Not measured on \(chip ?? "this Mac") yet" }

    static func runner(_ r: CatalogRunner, row: ModelRow, columns: [KindColumn], chip: String?) -> Runner {
        let results = r.results.compactMap { try? JSONDecoder().decode(ResultsFile.self, from: JSONEncoder().encode($0)) }
        let mode = row.fastAvailable && row.fast ? "fast" : "exact"
        let ordered = results.filter { $0.recordedAt != nil }.sorted(by: ResultsFile.newestFirst)
        let shown = ordered.first { $0.precision == row.selectedPrecision && $0.mode == mode } ?? ordered.first
        // One precision vocabulary across the window: "16-bit", as the Sizes line, the chip and the Models table say it.
        func bits(_ dtype: String) -> String { WeightsPresentation.bitLabel(dtype, formats: row.formats) }
        func figures(_ result: ResultsFile) -> [(String, String)] {
            let table = result.tableMetrics()
            return columns.compactMap { c in
                guard let value = table[c.id], value != MetricText.missing else { return nil }
                return (c.title, value.components(separatedBy: " \u{00b7} ")[0])
            }
        }
        func condition(_ result: ResultsFile) -> String {
            let profile = result.profile == "quick" ? "quick check" : result.profile == "full" ? "full run" : result.profile
            return [bits(result.precision), result.mode == "fast" ? "Fast" : "", profile, day(result.date)].filter { !$0.isEmpty }.joined(separator: " \u{00b7} ")
        }
        let help = results.map { res in ([bits(res.precision) + (res.mode == "fast" ? " Fast" : "")] + figures(res).map(\.1) + [condition(res)]).joined(separator: " \u{00b7} ")
        }
        return Runner(
            name: r.name, engine: engine(r.language), engineHelp: r.engine, isDefault: r.acceptedDefault, selected: r.selected, eligible: r.eligible,
            reason: r.eligible ? nil : r.ineligibleReason ?? "Not available on this Mac",
            figures: shown.map(figures) ?? [], figuresHelp: help.joined(separator: "\n"),
            condition: shown.map(condition) ?? notMeasured(chip))
    }

    /// `2 Oct` from an ISO date.
    static func day(_ iso: String) -> String {
        guard let date = ISO8601DateFormatter().date(from: iso) ?? ISO8601DateFormatter().date(from: iso + "T00:00:00Z") else { return "" }
        let f = DateFormatter(); f.locale = Locale(identifier: "en_GB"); f.dateFormat = "d MMM"
        return f.string(from: date)
    }
}
