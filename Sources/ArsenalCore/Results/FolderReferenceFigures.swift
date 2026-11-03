// SPDX-License-Identifier: MIT
import Foundation

/// Models-table figures for a local folder whose kind declares none (a kit kind such as local.document or local.forecast):
/// peak memory and the time of one warm request of the model's own operation, as quick checks.
///
/// None of it is ever written into the folder: every byte there is bound to the owner's code approval, so a figure file
/// added to it would ask for approval again. Two sources stand outside that digest:
/// - Arsenal's own Resources (`figures/folder-quick-checks.json`, written by scripts/import-quick-checks.py from a night's
///   measurements through the app), each entry bound to the model id, its kind, its weights revision, the precision and
///   the runner. A folder whose weights or runner differ gets none of them.
/// - The folder's own quick-check records (`results/quick-checks/<run>.json`, schema quick-check-1, as kit/new-model.py
///   writes them), already inside the approved digest; bound to the same weights revision.
/// Per precision the newest figure wins.
public struct FolderReferenceFigures: Sendable, Equatable {
    public static let path = "figures/folder-quick-checks.json"
    public static let schema = "folder-figures-1"

    public struct Entry: Codable, Sendable, Equatable {
        public var model: String
        public var kind: String
        public var revision: String
        public var precision: String
        public var runner: String
        public var peakMemoryGB: StatValue
        public var secondsPerRequest: StatValue
        public init(model: String, kind: String, revision: String, precision: String, runner: String, peakMemoryGB: StatValue, secondsPerRequest: StatValue) {
            self.model = model; self.kind = kind; self.revision = revision; self.precision = precision; self.runner = runner
            self.peakMemoryGB = peakMemoryGB; self.secondsPerRequest = secondsPerRequest
        }
        /// The newer of the entry's figures (an entry with no dated figure sorts last).
        var date: Date? { [peakMemoryGB, secondsPerRequest].compactMap { $0.provenance.flatMap { FolderReferenceFigures.instant($0.date) } }.max() }
    }
    private struct File: Codable { var schema: String; var entries: [Entry] }

    public var entries: [Entry]
    public init(entries: [Entry] = []) { self.entries = entries.compactMap(Self.valid) }

    /// The shipped figures, or none when the file is missing or unreadable (a missing figure is a triangle, never an error).
    public static func load(resources: URL) -> Self {
        guard let data = try? Data(contentsOf: resources.appendingPathComponent(path)), let figures = try? decode(data) else { return Self() }
        return figures
    }
    public static func decode(_ data: Data) throws -> Self {
        let file = try JSONDecoder().decode(File.self, from: data)
        guard file.schema == schema else { throw ManifestError.invalid("Unknown folder figures schema \(file.schema)") }
        return Self(entries: file.entries)
    }

    /// This folder's figures by precision: the shipped entries and the folder's own records that match its id, kind,
    /// weights revision and runner; per precision the newest.
    public func figures(_ model: ModelManifest, runner: String, records: [Entry] = []) -> [String: Entry] {
        var out: [String: Entry] = [:]
        for entry in entries + records.compactMap(Self.valid)
        where entry.model == model.id && entry.kind == model.kind && entry.revision == model.weights.revision && entry.runner == runner
            && model.precisions.contains(entry.precision)
        {
            if let held = out[entry.precision], (held.date ?? .distantPast) >= (entry.date ?? .distantPast) { continue }
            out[entry.precision] = entry
        }
        return out
    }

    /// A folder's own quick-check records (`results/quick-checks/*.json` among its evidence files) as entries: per
    /// precision row, peak memory (phys_footprint in MiB, stored in GB) and the warm time of one completed request
    /// (`timedOperation`). A record for other weights is ignored. `runner`: the folder's default runner, which its own
    /// records measured.
    public static func records(_ files: [String: Data], model: ModelManifest, runner: String, kind: KindSpec?) -> [Entry] {
        files.keys.sorted().filter { $0.hasPrefix(QuickCheckEvidence.directory) && $0.hasSuffix(".json") }.flatMap { path -> [Entry] in
            guard let data = files[path], let record = try? WireJSON.decode(data), record["schema"]?.string == QuickCheckEvidence.schema,
                record["model"]?.string == model.id, record["revision"]?.string == model.weights.revision, let chip = record["chip"]?.string, !chip.isEmpty
            else { return [] }
            let cores = record["gpu_cores"]?.number.map { Int($0) }
            let method = record["method"]?.string ?? "A quick check recorded in the folder."
            return (record["rows"]?.array ?? []).compactMap { row -> Entry? in
                guard let precision = row["precision"]?.string, let date = row["date"]?.string, instant(date) != nil else { return nil }
                let provenance = StatProvenance(
                    chip: row["chip"]?.string ?? chip, date: date, method: method, source: path + "; " + precision, revision: model.weights.revision,
                    run: record["run"]?.string, gpuCores: cores)
                var peak = StatValue()
                if let mib = row["peak_footprint_mb"]?.number, mib.isFinite, mib > 0 {
                    peak = StatValue(
                        state: .quickCheck, value: QuickCheckEvidence.gigabytes(mebibytes: mib), provenance: provenance, basis: "Recorded in the folder's own quick check.")
                }
                var request = StatValue()
                let ops = (row["ops"]?.object ?? [:]).filter { $0.value["status"]?.number == 200 && $0.value["seconds_warm"]?.number != nil }
                let chosen = timedOperation(completed: Set(ops.keys), kind: kind)
                if let op = chosen, let seconds = ops[op]?["seconds_warm"]?.number, seconds.isFinite, seconds >= 0 {
                    let title = kind?.operations.first { $0.name == op }?.title ?? op.replacingOccurrences(of: "_", with: " ").capitalized
                    request = StatValue(state: .quickCheck, value: seconds, provenance: provenance, basis: "One warm \(title) request from the folder's own quick check.")
                }
                guard peak.value != nil || request.value != nil else { return nil }
                return Entry(
                    model: model.id, kind: model.kind, revision: model.weights.revision, precision: precision, runner: runner, peakMemoryGB: peak,
                    secondsPerRequest: request)
            }
        }
    }

    /// The operation whose time stands for the model, the same rule scripts/import-quick-checks.py follows: the kind's
    /// primary operation, then its default operations, then its other declared operations, then the first completed by
    /// name.
    static func timedOperation(completed: Set<String>, kind: KindSpec?) -> String? {
        let declared = kind?.operations ?? []
        let order = (kind?.primaryOperation.map { [$0] } ?? []) + declared.filter(\.isDefault).map(\.name) + declared.filter { !$0.isDefault }.map(\.name)
        return order.first { completed.contains($0) } ?? completed.sorted().first
    }
    /// An entry with each figure checked: a number needs a quick-check or measured state, a finite non-negative value
    /// and a dated provenance with a chip; anything else reads as not measured. An entry left with no figure is dropped.
    static func valid(_ entry: Entry) -> Entry? {
        func checked(_ stat: StatValue) -> StatValue {
            guard [.quickCheck, .measured].contains(stat.state), let v = stat.value, v.isFinite, v >= 0, let p = stat.provenance, !p.chip.isEmpty, instant(p.date) != nil
            else { return StatValue() }
            return stat
        }
        var e = entry
        e.peakMemoryGB = checked(entry.peakMemoryGB); e.secondsPerRequest = checked(entry.secondsPerRequest)
        return e.peakMemoryGB.value == nil && e.secondsPerRequest.value == nil ? nil : e
    }
    static func instant(_ text: String) -> Date? {
        let plain = ISO8601DateFormatter(), fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return plain.date(from: text) ?? fractional.date(from: text)
    }
}
