// SPDX-License-Identifier: MIT
import Foundation
import MLX
import ArsenalKernelsCore

/// The kernels a runner received in `load.config.kernels`, one per `replaces` site, with their state for status.
///
/// Lifecycle per site: loaded (sources read, ASCII, digested) → the runner's gate decides (`activate` after the Exact
/// check or the Fast gate passed, `deactivate` with the reason otherwise) → a call that fails its compile preflight
/// returns nil and the site reports the compile failure. Sites start inactive: nothing serves before its gate.
public final class FolderKernels: @unchecked Sendable {
    /// nil: the supervisor sent no `config.kernels` (runner keeps its built-in behaviour).
    public let specs: [KernelSpec]?
    private let lock = NSLock()
    private var kernels: [String: FolderMLXKernel] = [:]
    private var notes: [String: (active: Bool, reason: String)] = [:]

    public var configured: Bool { specs != nil }

    public init(config value: Any?, compiler: KernelCompiler = .shared, verdicts: KernelVerdicts = .shared) throws {
        specs = try KernelSpec.parse(config: value)
        for spec in specs ?? [] {
            guard spec.api == .mlx else { notes[spec.replaces] = (false, "api \(spec.api.rawValue) kernels are not run by this MLX runner"); continue }
            do {
                kernels[spec.replaces] = try FolderMLXKernel(try LoadedKernel.load(spec), compiler: compiler, verdicts: verdicts)
                notes[spec.replaces] = (false, "not checked yet")
            } catch {
                notes[spec.replaces] = (false, "\(error)")
            }
        }
    }

    /// The declaration for a site (also when it is not runnable), or nil when the folder does not replace it.
    public func spec(_ replaces: String) -> KernelSpec? { specs?.first { $0.replaces == replaces } }

    /// The kernel for a site regardless of its gate state (the gate itself calls it).
    public func candidate(_ replaces: String) -> FolderMLXKernel? {
        lock.lock(); defer { lock.unlock() }
        return kernels[replaces]
    }

    /// The kernel for a site only while it is active and its compile preflight has not failed.
    public func active(_ replaces: String) -> FolderMLXKernel? {
        lock.lock(); defer { lock.unlock() }
        guard notes[replaces]?.active == true, let kernel = kernels[replaces], kernel.failure == nil else { return nil }
        return kernel
    }

    public func activate(_ replaces: String, reason: String) { set(replaces, true, reason) }
    public func deactivate(_ replaces: String, reason: String) { set(replaces, false, reason) }
    private func set(_ replaces: String, _ active: Bool, _ reason: String) {
        lock.lock(); defer { lock.unlock() }
        guard notes[replaces] != nil else { return }
        notes[replaces] = (active && kernels[replaces] != nil, reason)
    }

    public var states: [KernelState] {
        lock.lock(); defer { lock.unlock() }
        return (specs ?? []).map { spec in
            let note = notes[spec.replaces] ?? (false, "not loaded")
            if let failure = kernels[spec.replaces]?.failure {
                return KernelState(id: spec.id, replaces: spec.replaces, mode: spec.mode, active: false, reason: failure)
            }
            return KernelState(id: spec.id, replaces: spec.replaces, mode: spec.mode, active: note.active, reason: note.reason)
        }
    }

    /// `kernels` for the load reply and status.
    public var wire: [[String: Any]] { states.map(\.wire) }
}

/// The Exact gate: a model's outputs with the folder kernels on must equal its outputs on the stock path bit for bit,
/// on this chip, for the runner's fixed check inputs (the coverage is those inputs: the detail lists the output shapes
/// and dtypes it compared). Every kernel must have been dispatched during the "on" run and none may have failed its
/// preflight, or the verdict is a failure (`KernelVerdicts.gate`). Persisted by (kernel identities, model, dtype,
/// weights, the runner's check revision, executable identity, OS build, GPU); a passing verdict makes the next load of
/// the same build free.
public enum ExactGate {
    public struct Result: Sendable {
        public let pass: Bool
        public let detail: String
        public let persisted: Bool
    }

    /// `outputs(on)` returns the evaluated outputs of the runner's check inputs with the kernels on (true) or off.
    /// `weights`: the identity of the weights the check runs on when its outputs depend on them ("-" when the check
    /// uses synthetic inputs only).
    public static func qualify(
        kernels: [FolderMLXKernel], model: String, dtype: String, weights: String, checkRevision: String,
        verdicts: KernelVerdicts = .shared, outputs: (Bool) -> [MLXArray]
    ) -> Result {
        let identities = kernels.map(\.loaded.digest).sorted().joined(separator: "+")
        let key = KernelVerdicts.key("exact", identities, model, dtype, weights, checkRevision, KernelHost.binary)
        let (verdict, persisted) = verdicts.gate(key, kernels: kernels) {
            let stock = outputs(false), folder = outputs(true)
            guard stock.count == folder.count, !stock.isEmpty else { return .init(pass: false, detail: "check produced \(folder.count) vs \(stock.count) outputs") }
            var elements = 0
            for (i, (a, b)) in zip(folder, stock).enumerated() {
                let comparison = ExactComparison.compare(a, b)
                guard comparison.identical else { return .init(pass: false, detail: "output \(i): \(comparison)") }
                elements += comparison.elements
            }
            return .init(pass: true, detail: "bit-identical to the stock path on \(stock.count) check outputs, \(elements) elements (\(coverage(stock)))")
        }
        return Result(pass: verdict.pass, detail: verdict.detail, persisted: persisted)
    }

    /// Distinct output shapes and dtypes, in first-seen order, capped so the status line stays one sentence.
    static func coverage(_ outputs: [MLXArray]) -> String {
        var seen: [String] = []
        for o in outputs {
            let item = "\(o.dtype) " + o.shape.map(String.init).joined(separator: "x")
            if !seen.contains(item) { seen.append(item) }
        }
        let shown = seen.prefix(6).joined(separator: ", ")
        return seen.count > 6 ? shown + ", +\(seen.count - 6) more" : shown
    }
}
