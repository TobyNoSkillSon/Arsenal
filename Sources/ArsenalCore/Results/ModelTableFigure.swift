// SPDX-License-Identifier: MIT
import Foundation

extension StatProvenance {
    public var chipName: String {
        chip.split(separator: "-").enumerated().map { index, part in
            index == 0 ? part.uppercased() : part.prefix(1).uppercased() + part.dropFirst()
        }.joined(separator: " ")
    }
    /// Matching requires a known GPU configuration on both sides, even on the same chip model.
    public func isHost(chip hostChip: String = HostChip.current, gpuCores hostCores: Int = HostChip.gpuCores) -> Bool {
        guard let gpuCores, gpuCores > 0, hostCores > 0 else { return false }
        return chip.lowercased() == hostChip.lowercased() && gpuCores == hostCores
    }
    public var hardwareDescription: String { chipName + (gpuCores.map { " · \($0) GPU cores" } ?? "") }
}

/// A number and its unit are separate, with the original metric and evidence available for hover text.
public struct ModelTableFigure: Sendable, Equatable {
    public var value: Double?
    public var unit: String
    public var metric: StatMetric
    public var stat: StatValue
    /// The Peak memory cell and memory warnings print the same number at the same precision, in Activity Monitor's units.
    public static func memoryText(_ gigabytes: Double) -> String { MemoryDisplay.text(decimalGB: gigabytes) }
    public var isThisMac: Bool { stat.provenance?.isHost() == true }
    public var referenceChip: String? { isThisMac ? nil : stat.provenance?.chipName }
    init(metric: StatMetric, stat: StatValue, unit: String? = nil, reciprocal: Bool = false) {
        self.metric = metric; self.stat = stat; self.unit = unit ?? metric.unit
        if [.measured, .quickCheck, .derived, .declared].contains(stat.state), let number = stat.value,
            number.isFinite, number >= 0, !reciprocal || number > 0
        {
            value = reciprocal ? 1 / number : number
        } else {
            value = nil
        }
    }
}
/// Mac memory as the app shows it. Activity Monitor reports memory in binary GB (GiB, 2^30 bytes), so "100 GB" on a
/// 128 GB Mac is 100 GiB. Wire fields and stored figures are decimal GB (10^9 bytes): convert here, at display only.
public enum MemoryDisplay {
    public static let bytesPerDisplayGB = 1_073_741_824.0
    /// Decimal GB as Activity Monitor's GB.
    public static func gib(decimalGB: Double) -> Double { decimalGB * 1e9 / bytesPerDisplayGB }
    /// "5.62 GB" under 10, "80.9 GB" under 100, "100 GB" and "128 GB" from 100 up.
    public static func text(decimalGB: Double) -> String {
        let value = gib(decimalGB: decimalGB)
        let locale = Locale(identifier: "en_US_POSIX")
        if value < 9.995 { return String(format: "%.2f GB", locale: locale, value) }
        let tenths = (value * 10).rounded() / 10
        if tenths < 100 { return String(format: "%.1f GB", locale: locale, tenths) }
        return String(format: "%.0f GB", locale: locale, value.rounded())
    }
}
extension PrecisionStatSet {
    public var speed: ModelTableFigure {
        let metric: StatMetric
        switch kind {
        case .decisions: metric = .millisecondsPerDecision
        case .chat: metric = .tokensPerSecond
        case .images: metric = .secondsPerImage
        case .voices, .sound, .enhancement: metric = .realTimeFactor
        case .audio, .embeddings, .perception, .restoration, .geometry: metric = .secondsPerRequest
        }
        let stat = availability.state == .unavailable ? availability : value(metric)
        return ModelTableFigure(metric: metric, stat: stat, unit: metric == .realTimeFactor ? "× real time" : nil, reciprocal: metric == .realTimeFactor)
    }
    public var peakMemory: ModelTableFigure {
        let stat = availability.state == .unavailable ? availability : value(.peakMemoryGB)
        var figure = ModelTableFigure(metric: .peakMemoryGB, stat: stat, unit: "GB")
        if ![.measured, .quickCheck, .derived].contains(stat.state) { figure.value = nil }
        return figure
    }
}
