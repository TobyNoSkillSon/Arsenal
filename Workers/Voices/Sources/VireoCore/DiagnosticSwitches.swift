import Foundation

/// Environment switches the helper reads. Settings the app passes (and documented overrides) are normal operation;
/// every other `VIREO_*` variable changes behaviour for a test, lab run or diagnosis and is listed in /status
/// `test_hooks`, so a release never runs one silently. New switches are covered by the prefix rule; the contract
/// test `testEveryEnvironmentSwitchIsClassified` fails when a source file reads one that is in neither list.
public enum DiagnosticSwitches {
    /// Settings: the app's helper environment (Backend.helperEnvironment), `VIREO_PRECISION` and `VIREO_PORT` for
    /// harnesses, and CLI/app-side variables that never reach the helper's behaviour.
    public static let settings: Set<String> = [
        "VIREO_SUPPORT_DIR", "VIREO_PRELOAD", "VIREO_MANUAL_IDLE_MINUTES", "VIREO_ON_DEMAND_IDLE_MINUTES", "VIREO_ALLOW_SWAP",
        "VIREO_MUTED", "VIREO_CACHE_LIMIT_MB", "VIREO_DEFAULT_VOICE", "VIREO_KEEP", "VIREO_PRECISION", "VIREO_PORT",
        "VIREO_AGENT", "VIREO_NO_LAUNCH", "VIREO_HELPER", "VIREO_APP",
    ]
    /// Switches known to change behaviour (documentation; the prefix rule reports unknown ones too).
    public static let known: Set<String> = [
        "VIREO_STUB_MODELS", "VIREO_STUB", "VIREO_SILENT", "VIREO_KERNELS", "VIREO_REUSE_GATE_FAIL", "VIREO_REUSE_INJECT_NAN",
        "VIREO_SPEECH_SEED", "VIREO_QWEN_REF_CACHE", "VIREO_QWEN_PROFILE", "VIREO_QWEN_F32_SPEAKER", "VIREO_METALLIB",
        "VIREO_CATALOG", "VIREO_VOICES", "VIREO_BENCHMARKS", "VIREO_REFERENCES", "VIREO_DIAGNOSE_DIR", "VIREO_LOG_TO_STDERR",
        "VIREO_MIGRATE_FROM", "VIREO_TRACE_STATUS", "VIREO_RENDER_CHIP", "VIREO_TOOLTIP_GOLDEN",
        "VIREO_TEST_NOT_DOWNLOADED", "VIREO_TEST_MINUTE_SECONDS", "VIREO_TEST_MEMORY_FILE", "VIREO_TEST_VM_STATS",
        "VIREO_TEST_LOAD_FAULT", "VIREO_TEST_STUB_DELAY_MS", "VIREO_TEST_MAX_CONNECTIONS", "VIREO_TEST_READ_DEADLINE_S",
        "VIREO_UPDATE_API_URL", "VIREO_RELEASE_BASE_URL", "VIREO_UPDATE_CA_CERT", "VIREO_UPDATE_READY_SECONDS",
        "VIREO_TEST_INSTALL_STOP", "VIREO_TEST_TRANSCRIPT", "VIREO_TEST_TRANSCRIPT_HOLD",
        "VIREO_FORCE_STOCK", "VIREO_BREEZE_LAZY", "VIREO_BREEZE_CFG_BATCH", "VIREO_BREEZE_STREAM", "VIREO_BREEZE_PROFILE",
        "VIREO_BREEZE_STREAM_FRAMES", "VIREO_STREAM_BIAS_FIX", "VIREO_BREEZE_DEPTH_CACHE", "VIREO_BREEZE_FUSED", "VIREO_BREEZE_SMALLM",
        "VIREO_SA3_PROFILE", "VIREO_SA3_F32_WEIGHTS", "VIREO_SA3_HALF", "VIREO_SA3_SWA_VIEWS", "VIREO_SA3_INJECT_NAN", "VIREO_QWEN_FAST_PIPELINE",
        "VIREO_QWEN_FAST_ROPE", "VIREO_QWEN_FAST_TABLES", "VIREO_QWEN_FAST_SAMPLER",
        "VIREO_KOKORO_ALIGN", "VIREO_KOKORO_CONV_CACHE", "VIREO_KOKORO_LSTM", "VIREO_KOKORO_HALF",
        "VIREO_KOKORO_WARMUP", "VIREO_KOKORO_INJECT_NAN", "VIREO_KOKORO_PROFILE",
        "VIREO_SAM_HALF", "VIREO_SAM_SDPA", "VIREO_SAM_INJECT_NAN", "VIREO_SAM_GATE_FAIL", "VIREO_SAM_PROFILE",
        "VIREO_EXACT", "VIREO_REUSE_METAL",
    ]

    public static func isDiagnostic(_ name: String) -> Bool { name.hasPrefix("VIREO_") && !settings.contains(name) }

    /// Active diagnostic switches, sorted. `bundledCatalog`: the app sets VIREO_CATALOG to its own models.json, which
    /// is normal operation; any other catalog is reported.
    public static func active(in environment: [String: String], bundledCatalog: String? = nil) -> [String] {
        environment.keys.filter { name in
            guard isDiagnostic(name) else { return false }
            if name == "VIREO_CATALOG", let bundled = bundledCatalog, environment[name].map(standardized) == standardized(bundled) { return false }
            return true
        }.sorted()
    }
    static func standardized(_ path: String) -> String { URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path }

    // MARK: Model path (fast-path verdicts) and reference recording

    /// Families of switches that change which optimized components run.
    public static let modelPathPrefixes = ["VIREO_REUSE_", "VIREO_BREEZE_", "VIREO_QWEN_FAST_", "VIREO_SA3_", "VIREO_POCKET_", "VIREO_KOKORO_", "VIREO_SAM_"]
    /// Other switches that change what a model computes (stock/fast choice, codec fix, dtype, shaders, seed, cache).
    public static let modelPath: Set<String> = [
        "VIREO_KERNELS", "VIREO_FORCE_STOCK", "VIREO_STREAM_BIAS_FIX", "VIREO_QWEN_F32_SPEAKER", "VIREO_QWEN_REF_CACHE",
        "VIREO_METALLIB", "VIREO_SPEECH_SEED", "VIREO_EXACT",
    ]
    /// Known switches that leave model computation alone (besides every VIREO_TEST_* and *_PROFILE switch).
    public static let modelNeutral: Set<String> = [
        "VIREO_STUB_MODELS", "VIREO_STUB", "VIREO_SILENT", "VIREO_CATALOG", "VIREO_VOICES", "VIREO_BENCHMARKS", "VIREO_REFERENCES",
        "VIREO_DIAGNOSE_DIR", "VIREO_LOG_TO_STDERR", "VIREO_MIGRATE_FROM", "VIREO_TRACE_STATUS", "VIREO_RENDER_CHIP", "VIREO_TOOLTIP_GOLDEN",
        "VIREO_UPDATE_API_URL", "VIREO_RELEASE_BASE_URL", "VIREO_UPDATE_CA_CERT", "VIREO_UPDATE_READY_SECONDS",
    ]
    /// Part of every fast-path verdict key (VireoEngine `FastPath.identity`, the one configuration engines select their
    /// path from and the only way engine code reads these switches): a verdict reached while one is set is stored under a different
    /// key, so it can neither qualify nor disqualify the shipping path. Profiling switches only add timing lines.
    public static func affectsModelPath(_ name: String) -> Bool {
        !name.hasSuffix("_PROFILE") && (modelPath.contains(name) || modelPathPrefixes.contains { name.hasPrefix($0) })
    }
    /// The only diagnostic switches a reference recording tolerates: no playback, log lines on stderr.
    public static let allowedWhileRecording: Set<String> = ["VIREO_SILENT", "VIREO_LOG_TO_STDERR"]
    /// Active switches (`active(in:)`, the /status `test_hooks`) that make a reference recording refuse: references
    /// come only from the shipping path (bundled catalog, shaders and inputs, gates on, nothing forced).
    public static func recordingBlockers(_ active: [String]) -> [String] {
        active.filter { !allowedWhileRecording.contains($0) }.sorted()
    }
}

