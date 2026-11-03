// SPDX-License-Identifier: MIT
import Foundation
import VireoCore

/// The model-path configuration: every `VIREO_*` switch that changes what a model computes (fast-path choices, the
/// stock/fast selection, the codec bias fix, dtypes, shaders, seeds), in one typed value. Engines select their path
/// from it and `KernelGate` forms the self-test verdict identity from the same value, so a verdict reached under one
/// configuration never serves another. Only names `DiagnosticSwitches.affectsModelPath` accepts can be read here
/// (anything else traps), and engine code reads no other `VIREO_*` variable except `_PROFILE` timing switches
/// (contract test `testEngineReadsSwitchesOnlyThroughTheConfiguration`).
///
/// Each gated optimisation has its own switch, on by default; "0" turns it off for an A/B without a rebuild.
/// `VIREO_FORCE_STOCK=1` (or `VIREO_KERNELS=off`) turns every one off. No switch serves a path without its self-test.
public struct FastPath: Sendable, Equatable {
    /// The model-path switches that are set, by name.
    public let switches: [String: String]

    public init(environment: [String: String] = ProcessInfo.processInfo.environment, overrides: [String: String] = [:]) {
        switches = environment.merging(overrides) { _, cell in cell }.filter { DiagnosticSwitches.affectsModelPath($0.key) }
    }
    /// The configuration of this process now (no cell).
    public static var current: FastPath { FastPath() }
    /// One load's configuration: the process environment plus the selected cell's switches (`ModelSource.overrides`,
    /// `FastPathComponents.overrides`). Standard forces stock; Exact turns every inexact component off and sets
    /// `VIREO_EXACT=1`; Fast adds nothing. The verdict key follows, since it is this value's identity.
    public init(source: ModelSource, environment: [String: String] = ProcessInfo.processInfo.environment) {
        self.init(environment: environment, overrides: source.overrides)
    }

    /// Exact mode: every component that runs must be bit-identical to stock; self-tests that otherwise accept a
    /// tolerance demand identity.
    public var exact: Bool { value(FastPathComponents.exactSwitch) == "1" }

    /// The verdict-key component: every set model-path switch, sorted ("" when none is set).
    public var identity: String { switches.keys.sorted().map { "\($0)=\(switches[$0]!)" }.joined(separator: " ") }

    public func value(_ name: String) -> String? {
        precondition(DiagnosticSwitches.affectsModelPath(name), "\(name) is read as a model-path switch; add it to DiagnosticSwitches.modelPath")
        return switches[name]
    }

    public var forceStock: Bool { value("VIREO_FORCE_STOCK") == "1" || value("VIREO_KERNELS")?.lowercased() == "off" }
    /// One optimisation's switch: on unless forced stock or set to "0".
    public func enabled(_ name: String) -> Bool { !forceStock && !(exact && ["VIREO_SA3_HALF", "VIREO_REUSE_METAL"].contains(name)) && value(name) != "0" }
    /// An optimisation that is off by default (lab candidates, still self-tested): on only when set to "1".
    public func optIn(_ name: String) -> Bool { !forceStock && value(name) == "1" }

    // Lab switches read by the vendored models (installed there by `applyToVendorModels`).
    /// Streaming codec decode adds the transposed-conv bias once (the fix); "0" restores upstream (lab A/B).
    public var streamBiasFix: Bool { value("VIREO_STREAM_BIAS_FIX") != "0" }
    /// Qwen3-TTS speaker embedding kept in fp32 (the old path; lab A/B).
    public var qwenF32Speaker: Bool { value("VIREO_QWEN_F32_SPEAKER") == "1" }
    /// Codec frames per streamed Breeze decode (lab), nil for the default.
    public var breezeStreamFrames: Int? { value("VIREO_BREEZE_STREAM_FRAMES").flatMap(Int.init).flatMap { $0 > 0 ? $0 : nil } }
    /// Qwen3-TTS reference conditioning cache (default on; "0" recomputes per request).
    public var qwenReferenceCache: Bool { value("VIREO_QWEN_REF_CACHE") != "0" }
    /// Lab parity seed for speech requests without their own seed.
    public var speechSeed: UInt64? { value("VIREO_SPEECH_SEED").flatMap { UInt64($0) } }

    /// The vendored models take their lab switches from this configuration, never from the environment.

    // Shorthands on the current configuration (engines without a gate of their own configuration value).
    public static var forceStock: Bool { current.forceStock }
    public static func enabled(_ name: String) -> Bool { current.enabled(name) }
    public static func optIn(_ name: String) -> Bool { current.optIn(name) }

    /// Opt-in stderr profile line per generation (timing only; not part of the configuration).
    public static func profiling(_ name: String) -> Bool {
        precondition(name.hasSuffix("_PROFILE"), "\(name) is not a profiling switch")
        return ProcessInfo.processInfo.environment[name] == "1"
    }
    /// Profile level (0 off, 1 the usual line, 2 lab splits that add host syncs); timing only, like `profiling`.
    public static func profilingLevel(_ name: String) -> Int {
        precondition(name.hasSuffix("_PROFILE"), "\(name) is not a profiling switch")
        return ProcessInfo.processInfo.environment[name].flatMap { Int($0) } ?? 0
    }
}
