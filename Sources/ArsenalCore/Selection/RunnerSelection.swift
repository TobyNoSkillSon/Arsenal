// SPDX-License-Identifier: MIT
import Foundation

public struct RunnerSelectionEnvironment: Sendable {
    public var chip: String; public var macOS: String; public var installedWorkers: Set<String>
    public init(chip: String, macOS: String, installedWorkers: Set<String>) {
        self.chip = chip; self.macOS = macOS; self.installedWorkers = installedWorkers
    }
}
public struct RunnerChoice: Codable, Sendable, Equatable {
    public var runner: String; public var reason: String
    public init(runner: String, reason: String) { self.runner = runner; self.reason = reason }
}
public actor RunnerSelector {
    public init(support: URL) {}
    public static func eligible(_ runner: RunnerSpec, environment: RunnerSelectionEnvironment, fast: Bool) -> Bool {
        guard runner.chips.contains("*") || runner.chips.contains(RunnerSelection.family(environment.chip)), version(environment.macOS, atLeast: runner.minMacOS),
            !fast || runner.fastExact
        else { return false }
        guard runner.language != "python" else { return false }
        return runner.worker.map { environment.installedWorkers.contains($0) } ?? false
    }
    public func select(model: ModelManifest, environment: RunnerSelectionEnvironment, fast: Bool) throws -> RunnerChoice {
        let name = model.selection?.runner(chip: environment.chip) ?? model.runners.first?.name
        guard let runner = model.runners.first(where: { $0.name == name }), Self.eligible(runner, environment: environment, fast: fast) else {
            throw ManifestError.invalid("accepted default runner is unavailable or incompatible for \(model.id)")
        }
        return RunnerChoice(runner: runner.name, reason: "accepted folder default for \(RunnerSelection.family(environment.chip)); no on-device ranking")
    }
    private static func version(_ current: String, atLeast minimum: String) -> Bool {
        let a = current.split(separator: ".").compactMap { Int($0) }; let b = minimum.split(separator: ".").compactMap { Int($0) }
        guard !a.isEmpty && !b.isEmpty else { return false }
        for i in 0..<max(a.count, b.count) { let x = i < a.count ? a[i] : 0; let y = i < b.count ? b[i] : 0; if x != y { return x > y } }; return true
    }
}
