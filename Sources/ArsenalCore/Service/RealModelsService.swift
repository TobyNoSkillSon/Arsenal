// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

public actor RealModelsService: ModelsService {
    let activityAggregator = ActivityAggregator()
    public let paths: ArsenalPaths
    public let supervisor: Supervisor
    let installer: ModelFolderInstaller
    var source: any CatalogSource
    let artifacts: any WorkerArtifact
    let fetcher: ModelFetcher
    var catalog: CatalogDocument
    // Display/removal metadata only; withdrawn catalog models never become unsigned local models.
    var droppedCatalogModels: [String: DroppedCatalogModel] = [:]
    private var publicKey = Data()
    private var catalogDirectory: URL?
    private var bundledResources: URL?
    private var refreshing = false
    var localDiscoveryError: String?
    /// The app's watch on `local-models/` (none in tests and headless commands).
    var localFolderWatcher: LocalFolderWatcher?
    var localRescanRunning = false, localRescanWanted = false
    private var configData: Data?
    private var persistedConfig: ArsenalConfiguration
    var residentMenus: [String: [LoadedMenuSection]] = [:]
    var residentOperations: [String: [ModelOperation]] = [:]
    var residentSettings: [String: ModelSettings] = [:]
    var reloading = Set<String>()
    var residentLaunches: [String: WorkerLaunch] = [:]
    var fingerprintCache: [String: String] = [:]
    var dormantRevisions: [String: ResidentRevision] = [:]
    var previousRevisions: [String: ResidentRevision] = [:]
    var rejectedRevisionDigests: [String: String] = [:]
    var swapTasks: [String: Task<Void, Never>] = [:]
    var residentLoads: [String: JSONObject] = [:]
    var residentMemoryRequirements: [String: LoadMemoryRequirement] = [:]
    var residentManifests: [String: ModelManifest] = [:]
    var runnerResultsCache: [String: [String: [JSONObject]]] = [:]
    var historyCache: [String: [String: RunnerHistory]] = [:]
    var loadedRunners: [String: String] = [:]
    var precisionResultCache: [String: [PrecisionStatsResult]] = [:]
    var catalogKernelManifests: [String: Data] = [:]
    var resultCache: [String: [ResultsFile]] = [:]
    /// Figures Arsenal ships for local folders whose kind declares none (Resources/figures), and each such folder's own
    /// quick-check records, by model id (FolderReferenceFigures).
    var folderFigures = FolderReferenceFigures()
    var folderFigureRecords: [String: [FolderReferenceFigures.Entry]] = [:]
    var specs: [KindSpec]
    var config: ArsenalConfiguration
    private let localScanner = LocalFolderScanner()
    /// Local folders' pin checks against Hugging Face (`RealModelsService+PinnedFiles`); off until the app sets a lister.
    var pinnedFiles = PinnedFilesState()
    private var localEngineProblems: [String: String] = [:]
    private(set) var localProblemIDs = Set<String>()
    private(set) var brokenLocalFolders: [String: String] = [:]
    var localModels: [String: LocalModelFolder] = [:]
    private(set) var lastLocalDiscovery: LocalFolderDiscovery?
    var kindsDirectory: URL?
    var operationCache: [String: [ModelOperation]] = [:]
    /// A failed schema resolution, keyed by what would change it (catalog commit, local folder digest, kinds directory), so a
    /// persistent failure is read once per change, not once per caller or refresh.
    var operationFailures: [String: (key: String, error: any Error)] = [:]
    private var settingsWarnings: [String: String] = [:]
    var modelSettings = ModelSettings()
    var metadata: [String: JSONObject] = [:]
    var modelRows: [String: ModelRow] = [:]
    var loadedPreferences: [String: ModelPreferences] = [:]
    var residentLocalDigests: [String: String] = [:]
    var residents: [String: ResidentModel] = [:]
    /// Consent belongs to this resident's lifetime, never to a saved setting or a later Load.
    var forceLoadedResidents: Set<String> = []
    /// Each open live session's one Activity entry (ServiceActivity), by model, operation and session ID.
    var sessionActivities: [String: (activity: UUID, seen: Date)] = [:]
    var footprints: [String: WorkerMemory] = [:]
    var loadReservations: [String: Double] = [:]
    var loadReservationReleases: Set<String> = []
    var loadTasks: [String: Task<Void, any Error>] = [:]
    var unloading: [String: Int] = [:]
    var updating = false
    var memoryWaiters: [UUID: CheckedContinuation<Void, any Error>] = [:]
    var pendingRuns: [String: Int] = [:]
    var waitingRevisionRuns: [String: Int] = [:]
    var memoryAdmissionOrder: [UUID] = []
    var memoryPeriodicWaiters = Set<UUID>()
    var memoryRecheckTask: Task<Void, Never>?
    let memoryRecheckSeconds: Double
    var localCodeCheckBarrier: (@Sendable () async -> Void)?
    private var getPrecisions: [String: String] = [:]
    var shuttingDown = false
    private var shutdownTask: Task<Void, Never>?
    var runTasks: [UUID: Task<JSONObject, any Error>] = [:]
    private var getReplans: Set<String> = []
    private var getReplanPrecision: [String: String] = [:]
    private var getProgress: [String: GetProgressAccumulator] = [:]
    var gets: [String: Task<Void, any Error>] = [:]
    /// Gets waiting on a gated Hugging Face model (RealModelsService+Gate.swift).
    var gateRetry = GateRetry()
    /// Opens Hugging Face pages in the browser; the app sets it (NSWorkspace). Nothing opens in tests or headless use.
    var openPages: @Sendable ([URL]) -> Void = { _ in }
    var timers: [String: Task<Void, Never>] = [:]
    private var listeners: [UUID: AsyncStream<ServiceEvent>.Continuation] = [:]
    let probe: MemoryProbe
    private let minuteSeconds: Double
    private let sourceRetentionThresholdBytes: Int64
    private var pressure: MemoryPressureMonitor?
    private var sourceName: String
    private var sourceError: String?
    private var installationUpdates: [String: Set<String>] = [:]
    private var installedRowCache: [String: InstalledRowCacheKey] = [:]
    /// Whether each precision's snapshot verifies, keyed by its files' stamps: a corrupt or replaced file changes its stamp.
    var intactMemo: [String: (stamps: [InstalledRowCacheKey.File], intact: Bool)] = [:]
    var catalogEntryCache: [String: (key: CatalogEntryCacheKey, entry: CatalogEntry)] = [:]

    private init(
        paths: ArsenalPaths, catalog: CatalogDocument, installer: ModelFolderInstaller, source: any CatalogSource, sourceName: String, kinds: [KindSpec],
        artifacts: any WorkerArtifact, client: HuggingFaceClient, configuration: ArsenalConfiguration, minuteSeconds: Double, sourceRetentionThresholdBytes: Int64,
        memoryProbe: MemoryProbe, memoryRecheckSeconds: Double
    ) {
        self.paths = paths; self.catalog = catalog; self.installer = installer; self.source = source; self.sourceName = sourceName; specs = kinds
        self.artifacts = artifacts; supervisor = Supervisor(); probe = memoryProbe; config = configuration; persistedConfig = configuration; self.minuteSeconds = minuteSeconds
        self.sourceRetentionThresholdBytes = sourceRetentionThresholdBytes; self.memoryRecheckSeconds = memoryRecheckSeconds
        rejectedRevisionDigests = config.rejectedRevisions
        configData = try? Data(contentsOf: paths.support.appendingPathComponent("config.json"))
        fetcher = ModelFetcher(support: paths.support, client: client, localSources: configuration.get["localSources"]?.array?.compactMap(\.string))
    }
    public static func open(
        catalogData: Data, signature: Data, publicKey: Data, source: any CatalogSource, sourceName: String = "local", kinds: [KindSpec], paths: ArsenalPaths = ArsenalPaths(),
        artifacts: (any WorkerArtifact)? = nil, client: HuggingFaceClient = HuggingFaceClient(), minuteSeconds: Double = 60, sourceRetentionThresholdBytes: Int64 = 8_000_000_000,
        kindsDirectory: URL? = nil, memoryProbe: MemoryProbe = MemoryProbe(), memoryRecheckSeconds: Double = 30
    ) async throws -> RealModelsService {
        try paths.create()
        let suppliedKinds = kindsDirectory ?? (source as? BundleCatalogSource)?.kindsDirectory
        let installer = ModelFolderInstaller(support: paths.support, kindsDirectory: suppliedKinds)
        let catalog = try await installer.accept(data: catalogData, signature: signature, publicKey: publicKey)
        if let suppliedKinds, source is BundleCatalogSource, !(catalog.kindFiles ?? []).isEmpty { try catalog.verifyKinds(directory: suppliedKinds) }
        let verifiedKinds = try await installer.prepareKinds(source: source)
        let resolvedKinds = !(catalog.kindFiles ?? []).isEmpty && verifiedKinds != nil ? try KindSpec.load(directory: verifiedKinds!) : kinds
        var configuration = try ArsenalConfiguration.read(paths.support.appendingPathComponent("config.json"))
        let retireExact = configuration.selections.values.contains { !$0.fast }
        for id in configuration.selections.keys { configuration.selections[id]?.fast = true }
        let service = RealModelsService(
            paths: paths, catalog: catalog, installer: installer, source: source, sourceName: sourceName, kinds: resolvedKinds,
            artifacts: artifacts ?? BundledWorkerArtifact(), client: client, configuration: configuration, minuteSeconds: minuteSeconds,
            sourceRetentionThresholdBytes: sourceRetentionThresholdBytes, memoryProbe: memoryProbe, memoryRecheckSeconds: memoryRecheckSeconds)
        await service.setRefreshInputs(
            publicKey: publicKey,
            directory: (source as? BundleCatalogSource)?.resources.appendingPathComponent("catalog")
                ?? (source as? LocalGitCatalogSource)?.repo.appendingPathComponent("Resources/catalog"))
        await service.setKindsDirectory(verifiedKinds ?? suppliedKinds)
        try await service.initializeRows()
        if retireExact { try await service.persist() }
        await service.cleanStaging(); await service.collectWeights(startup: true); await service.startPressureMonitor(); return service
    }
    /// Reads the configured local/HTTPS source, or a bundled catalog, and always verifies the signature.
    public static func open(
        repository: URL, resources: URL? = nil, paths: ArsenalPaths = ArsenalPaths(), artifacts: (any WorkerArtifact)? = nil, explicitCatalogRepository: URL? = nil,
        sourceRetentionThresholdBytes: Int64 = 8_000_000_000
    ) async throws -> RealModelsService {
        let resources = resources ?? repository
        let cfg = try ArsenalConfiguration.read(paths.support.appendingPathComponent("config.json"))
        let setting = cfg.catalog["source"]?.object ?? [:]
        let bundle = try? CatalogBootstrap.verifiedBundleSource(resources: resources)
        let keyLocations = [
            resources.appendingPathComponent("catalog-public-key"), repository.appendingPathComponent("Resources/catalog-public-key"),
            resources.appendingPathComponent("Resources/catalog-public-key")
        ]
        guard let key = keyLocations.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw WorkerFailure(code: "catalog_key", message: "Catalog verification key is missing; reinstall Arsenal.")
        }
        let selected = try CatalogBootstrap.select(
            bundledDirectory: resources.appendingPathComponent("catalog"), cachedDirectory: paths.directory("catalog"), publicKey: Data(contentsOf: key))
        var source: any CatalogSource
        var sourceName: String
        var sourceError: String?
        var migrateSource = false
        let configuredPath = explicitCatalogRepository?.path ?? setting["path"]?.string
        if let path = configuredPath, explicitCatalogRepository == nil, setting["explicit"]?.bool != true,
            URL(fileURLWithPath: path).standardizedFileURL.path == LocalGitCatalogSource.defaultRepository.path, let bundle
        {
            source = bundle; sourceName = "bundle"; migrateSource = true
        } else if let path = configuredPath {
            let local = LocalGitCatalogSource(repo: URL(fileURLWithPath: path))
            source = local; sourceName = path
            do {
                try local.validateAvailability()
                if !(selected.document.kindFiles ?? []).isEmpty { try selected.document.verifyKinds(directory: local.repo.appendingPathComponent("Resources/kinds")) }
            } catch {
                guard let bundle else { throw error }
                source = bundle; sourceName = "bundle"; sourceError = "Using bundled catalog: \(error.localizedDescription)"
            }
        } else if let base = setting["rawBase"]?.string, let url = URL(string: base) {
            source = HTTPSCatalogSource(rawBase: url); sourceName = base
        } else {
            guard let bundle else { throw ManifestError.invalid("No verified bundled model catalog") }; source = bundle; sourceName = "bundle"
            migrateSource = cfg.catalog["source"] == nil
        }
        let kindDir = (source as? BundleCatalogSource)?.kindsDirectory
        let fallbackKinds = resources.appendingPathComponent("kinds")
        let service = try await open(
            catalogData: selected.data, signature: selected.signature, publicKey: Data(contentsOf: key), source: source, sourceName: sourceName,
            kinds: (try? KindSpec.load(directory: kindDir ?? fallbackKinds)) ?? [], paths: paths, artifacts: artifacts,
            sourceRetentionThresholdBytes: sourceRetentionThresholdBytes, kindsDirectory: kindDir)
        if migrateSource { try await service.migrateCatalogSource() }
        await service.setBundledResources(resources)
        await service.setSourceError(sourceError); return service
    }
    private func migrateCatalogSource() throws {
        config.catalog["source"] = "bundle"; config.migrations["catalogBundleV2"] = true
        try persist()
    }
    private func setKindsDirectory(_ directory: URL?) { kindsDirectory = directory }
    private func cleanStaging() async {
        do {
            let paths = self.paths
            _ = try await Task.detached {
                let count = try StagingCleanup.run(paths: paths); try TransientModelJobs.prune(support: paths.support)
                try? OwnedCacheCleanup.pruneReceipts(support: paths.support); return count
            }.value
        } catch { emit(.error(id: nil, message: "Staging cleanup skipped: \(error.localizedDescription)")) }
    }
    private func setSourceError(_ error: String?) { sourceError = error; emit(.catalog(catalogStatus())) }

    func setBundledResources(_ resources: URL) { bundledResources = resources; loadFolderFigures(resources) }
    private func setRefreshInputs(publicKey: Data, directory: URL?) { self.publicKey = publicKey; catalogDirectory = directory }
    private func initializeRows(preservingActive: Bool = false) async throws {
        modelSettings.load(config.settings)
        catalogKernelManifests.removeAll(keepingCapacity: true)
        for item in catalog.models {
            let launchFiles = item.files.filter { file in
                file.bytes <= 1_048_576
                    && ((file.path.hasPrefix("results/") && (file.path.hasSuffix(".json") || file.path.hasSuffix(".md")))
                        || (file.path.hasPrefix("log/") && file.path.hasSuffix(".md"))
                        || file.path == "kernels/manifest.toml" || file.path == SettingsPanel.fileName)
            }.map { "models/\(item.id)/\($0.path)" }
            let launchEvidence = await source.readAvailable(paths: launchFiles, commit: catalog.sourceCommit)
            let m = item.manifest
            metadata[m.id] = (try? WireJSON.decode(JSONEncoder().encode(m))) ?? [:]
            modelSettings.declare(m.id, metadata: metadata[m.id])
            let old = try? await installer.installedManifest(id: m.id)
            for precision in m.precisions {
                let snapshotChanged = (try? await fetcher.updateRequired(model: m, precision: precision)) == true
                let oldInstalled = old.map { FileManager.default.fileExists(atPath: StoredNames.snapshotURL(in: snapshot($0, precision)).path) } ?? false
                if (oldInstalled && old != nil && !m.hasSameInstallationPins(as: old!, precision: precision)) || snapshotChanged {
                    installationUpdates[m.id, default: []].insert(precision)
                }
            }
            var evidence: [String: Data] = [:]
            for file in item.files
            where (file.path.hasPrefix("results/") && (file.path.hasSuffix(".json") || file.path.hasSuffix(".md")) || file.path.hasPrefix("log/") && file.path.hasSuffix(".md"))
                && file.bytes <= 1_048_576
            {
                if let data = launchEvidence["models/\(m.id)/\(file.path)"], Int64(data.count) == file.bytes, FileDigest.sha256(data) == file.sha256 { evidence[file.path] = data }
            }
            cacheEvidence(m, files: evidence); metadata[m.id]?[SettingsPanel.metadataKey] = SettingsPanel.verified(item, files: launchEvidence)
            let p = config.selections[m.id] ?? ModelPreferences(precision: m.defaultPrecision)
            let folder = paths.directory("models").appendingPathComponent(m.id)
            // The row's effective mode: a runner without a separate Fast path runs Exact whatever the preference says.
            let rowFast = selectedRunner(m)?.fastExact ?? false
            let results = readResults(folder: folder).filter { $0.precision == p.precision && $0.mode == (rowFast ? "fast" : "exact") }
            let chip = Self.chip; let badge = MeasurementBadge.derive(results: results, chip: chip)
            let shown =
                results.first { $0.chip == chip && $0.eligibleMeasurement } ?? results.first { $0.chip == chip && $0.verificationLabel == "quick check" }
                ?? results.first { $0.eligibleMeasurement }
            var metrics: JSONObject = shown?.tableMetrics(columns: kinds().first { $0.id == m.kind }?.columns).mapValues { .string($0) } ?? [:]
            if let context = m.context { metrics["context"] = .number(Double(context)) }
            // A Get waiting on a gated model is active too: its row keeps the wait and the counter that goes with it.
            let gateWaiting = modelRows[m.id]?.states.values.contains { $0.phase == PrecisionState.gatedPhase } == true || gateRetry.isWaiting(m.id)
            let activeRow =
                preservingActive && (residents[m.id] != nil || gets[m.id] != nil || loadTasks[m.id] != nil || gateWaiting) ? modelRows[m.id] : nil
            modelRows[m.id] = ModelRow(
                id: m.id, name: m.name, kind: m.kind, params: m.params, licence: m.licence.spdx, precisions: m.precisions,
                selectedPrecision: servablePrecision(m, p.precision), fast: rowFast,
                runner: selectedRunner(m)?.name ?? "Unavailable", badge: ModelBadge(rawValue: badge.rawValue) ?? .unverified, metrics: metrics,
                peakRAMMB: shown?.metrics["peak_ram_mb"] ?? nil, keepHot: p.keepHot, sizeBytes: m.weights.bytes?.values.reduce(0, +), summary: m.summary,
                downloadBytes: m.downloadBytesByPrecision,
                verification: shown?.verificationLabel == "quick check" ? "quick check" : badge == .unverified ? "Not tried on this Mac yet" : badge.rawValue,
                publisher: m.publisher, released: m.released, commercial: m.licence.commercial, runnerLanguage: selectedRunner(m)?.language ?? "",
                fastExact: selectedRunner(m)?.fastExact ?? false,
                quickMetrics: shown?.quickCheck == true ? (shown?.tableMetrics() ?? [:]).filter { $0.key != "context" }.mapValues { _ in true } : [:])
            modelRows[m.id]?.purpose = ModelPurpose.text(m, kind: kinds().first { $0.id == m.kind })
            if let activeRow {
                modelRows[m.id]?.states = activeRow.states
                modelRows[m.id]?.residentRunner = activeRow.residentRunner
                modelRows[m.id]?.kernels = activeRow.kernels
                modelRows[m.id]?.approval = activeRow.approval
                // A removed dtype remains visible until its already-loaded runner exits.
                if let dtype = activeRow.residentDType, modelRows[m.id]?.precisions.contains(dtype) != true { modelRows[m.id]?.precisions.append(dtype) }
            }
            if let file = item.files.first(where: { $0.path == "kernels/manifest.toml" }), file.bytes <= 1_048_576,
                let data = launchEvidence["models/\(item.id)/kernels/manifest.toml"], Int64(data.count) == file.bytes, FileDigest.sha256(data) == file.sha256,
                let kernels = try? TOMLParser.decode(KernelManifest.self, from: String(decoding: data, as: UTF8.self))
            {
                catalogKernelManifests[item.id] = data
                modelRows[item.id]?.hasFastKernels = kernels.kernels.contains { $0.mode == "fast" && ($0.chip == RunnerSelection.family(Self.chip) || $0.chip == "*") }
                if modelRows[item.id]?.fastAvailable == true { modelRows[item.id]?.fast = true }
            }
            do { try enqueueMeasurements(m) } catch { emit(.error(id: m.id, message: error.localizedDescription)) }
        }
        if config.legacyTableNeedsMigration {
            config.table = catalog.models.filter { Self.hasInstalledSnapshot(id: $0.id, support: paths.support) }.map(\.id)
        }
        if config.migrations["tableV2"]?.bool != true {
            config.migrations["tableV2"] = true
            let migrated = config
            do { try persist() } catch {
                config = migrated // An unsaved migration must still show the user's installed models.
                emit(.error(id: nil, message: "Table migration could not be saved: \(error.localizedDescription)"))
            }
        }
        for item in catalog.models { updateRunnerRow(item.manifest); updatePending(item.id) }
        try await reconcileDroppedCatalogModels()
        await refreshLocalModels()
        await refreshInstalled(forceDownloadSizes: true)
    }
    var allManifests: [ModelManifest] { catalog.models.map(\.manifest) + localModels.values.map(\.manifest).sorted { $0.id < $1.id } }
    func discoverLocalModels(notifyChanges: Bool = true) async throws {
        kindsDirectory = try await installer.verifiedKindsDirectory()
        let report = applyPinnedFilesChecks( // a folder pinning a file its repository lacks is an error row
            try await localScanner.scan(support: paths.support, signedIDs: Set(catalog.models.map(\.id)), kinds: Set(specs.map(\.id)), kindsDirectory: kindsDirectory))
        if report.problems.isEmpty, brokenLocalFolders.isEmpty, report == lastLocalDiscovery {
            // Engine binaries live outside the folder inventory and can change independently.
            for local in report.folders { refreshLocalEngine(local, refreshConsent: false) }
            return
        }
        let discovered = report.folders
        for id in localProblemIDs { modelRows[id] = nil; installedRowCache[id] = nil; catalogEntryCache[id] = nil }
        localProblemIDs = []
        let previouslyBroken = Set(brokenLocalFolders.keys)
        brokenLocalFolders = [:]
        let discoveredIDs = Set(discovered.map { $0.manifest.id })
        let knownLocalIDs = Set(localModels.keys).union(
            config.table.filter { id in
                !catalog.models.contains { $0.id == id } && droppedCatalogModels[id] == nil
            })
        let absent = knownLocalIDs.subtracting(discoveredIDs).subtracting(report.problems.map(\.id)).subtracting(report.superseded)
        let problems = report.problems + absent.sorted().map { LocalFolderProblem(id: $0, message: "This local model folder is missing.") }
        for problem in problems {
            if !catalog.models.contains(where: { $0.id == problem.id }), config.table.contains(problem.id) || localModels[problem.id] != nil {
                let id = problem.id
                var row = modelRows[id] ?? ModelRow(id: id, name: id, kind: "local", precisions: ["bf16"], runner: "Unavailable", fastExact: false)
                row.local = true; row.verification = problem.message; row.needsCodeConsent = false
                row.states[row.selectedPrecision] = .init(state: .failed, error: problem.message, phase: "localFolderError")
                if let loaded = loadedPreferences[id]?.precision {
                    row.states[loaded] = .init(state: .failed, error: problem.message, phase: "localFolderErrorLoaded")
                }
                modelRows[id] = row; brokenLocalFolders[id] = problem.message
                // Discovery errors belong to this row; they must not replace the current action's footer error.
                continue
            }
            let id = "local-error-" + FileDigest.sha256(Data(problem.id.utf8)).prefix(12)
            var row = ModelRow(id: String(id), name: problem.id, kind: "local", precisions: [], runner: "Unavailable", fastExact: false)
            row.local = true; row.verification = problem.message; row.summary = problem.message; row.states["bf16"] = .init(state: .failed, error: problem.message)
            modelRows[row.id] = row; localProblemIDs.insert(row.id)
        }
        for local in discovered {
            let m = local.manifest, id = m.id
            if previouslyBroken.contains(id) {
                for dtype in modelRows[id]?.precisions ?? [] where modelRows[id]?.states[dtype]?.phase?.hasPrefix("localFolderError") == true {
                    modelRows[id]?.states[dtype] = .init(state: loadedPreferences[id]?.precision == dtype ? .loaded : .notInstalled)
                }
                modelRows[id]?.verification = "Local · unsigned folder"
            }
            let changedDigest = localModels[id]?.contentDigest != local.contentDigest
            if changedDigest {
                metadata[id] = try WireJSON.decode(JSONEncoder().encode(m))
                modelSettings.declare(id, metadata: metadata[id])
                metadata[id]?[SettingsPanel.metadataKey] = SettingsPanel.verified(local); operationCache[id] = nil; operationFailures[id] = nil
                let evidence = await Task.detached { () -> [String: Data] in
                    var files: [String: Data] = [:]
                    for path in local.fileDigests.keys
                    where (path.hasPrefix("results/") && (path.hasSuffix(".json") || path.hasSuffix(".md")) || path.hasPrefix("log/") && path.hasSuffix(".md")) {
                        if let url = try? SafePath.resolve(path, inside: local.directory), let size = try? FileDigest.bytes(url), size <= 1_048_576,
                            let data = try? Data(contentsOf: url), FileDigest.sha256(data) == local.fileDigests[path]
                        {
                            files[path] = data
                        }
                    }
                    return files
                }.value
                cacheEvidence(m, files: evidence, local: local)
                let preferences = config.selections[id] ?? ModelPreferences(precision: m.defaultPrecision)
                var row = modelRows[id] ?? ModelRow(id: id, name: m.name, kind: m.kind)
                row.name = m.name; row.kind = m.kind; row.summary = m.summary; row.params = m.params; row.licence = m.licence.spdx
                row.purpose = ModelPurpose.text(m, kind: kinds().first { $0.id == m.kind } ?? (try? EmbeddedKind.resolve(model: m, folder: local.directory))?.spec)
                row.publisher = m.publisher; row.released = m.released; row.commercial = m.licence.commercial; row.precisions = m.precisions
                row.selectedPrecision = servablePrecision(m, preferences.precision)
                row.fastExact = selectedRunner(m)?.fastExact ?? false; row.fast = row.fastExact
                row.keepHot = preferences.keepHot; row.runner = selectedRunner(m)?.name ?? "Unavailable"; row.runnerLanguage = selectedRunner(m)?.language ?? ""
                row.sizeBytes = m.weights.bytes?.values.reduce(0, +); row.downloadBytes = m.downloadBytesByPrecision
                row.local = true; row.contentDigest = local.contentDigest; row.verification = "Local · unsigned folder"
                row.hasFastKernels = try KernelManifest.loadIfPresent(folder: local.directory).kernels.contains {
                    $0.mode == "fast" && ($0.chip == RunnerSelection.family(Self.chip) || $0.chip == "*")
                }
                row.fast = row.fastAvailable
                modelRows[id] = row
            }
            // Keep the old digest if applying the newly scanned folder throws; the next refresh retries it.
            localModels[id] = local
            do { try enqueueMeasurements(local.manifest) } catch {
                modelRows[id]?.verification = error.localizedDescription
                if notifyChanges { emit(.error(id: id, message: error.localizedDescription)) }
            }
            if config.rejectedRevisions[id] == local.contentDigest {
                do { try await readRecoveryRevision(id) } catch {
                    let message = "Previous revision recovery: \(error.localizedDescription)"
                    modelRows[id]?.verification = message
                    if notifyChanges { emit(.error(id: id, message: message)) }
                }
            }
            refreshLocalEngine(local)
            updateRunnerRow(m)
            modelRows[id]?.approval = config.localCodeApprovals[id]
            updatePending(id)
        }
        let present = Set(discovered.map { $0.manifest.id }).union(brokenLocalFolders.keys)
        for id in Array(localModels.keys) where !present.contains(id) {
            if residents[id] != nil { try await unload(id: id, notifyChanges: notifyChanges) }
            localEngineProblems[id] = nil; localModels[id] = nil; operationCache[id] = nil; operationFailures[id] = nil; installedRowCache[id] = nil; catalogEntryCache[id] = nil
            forgetVanishedLocalFolder(id)
        }
        lastLocalDiscovery = report
        if notifyChanges, previouslyBroken != Set(brokenLocalFolders.keys) { changed() }
    }
    private func refreshLocalEngine(_ local: LocalModelFolder, refreshConsent: Bool = true) {
        let m = local.manifest, id = m.id, hadProblem = localEngineProblems[local.manifest.id] != nil
        do {
            if let runner = selectedRunner(m) { try artifacts.validateEngine(model: m, runner: runner) }
            if refreshConsent || hadProblem {
                modelRows[id]?.needsCodeConsent = (try Self.localCodeRequiresConsent(local)) && config.localCodeConsent[id] != local.contentDigest
            }
            localEngineProblems[id] = nil
            if hadProblem {
                modelRows[id]?.verification = "Local · unsigned folder"
                for dtype in modelRows[id]?.precisions ?? [] where modelRows[id]?.states[dtype]?.phase == "localEngineError" {
                    modelRows[id]?.states[dtype] = .init(state: loadedPreferences[id]?.precision == dtype ? .loaded : snapshotPresent(m, dtype) ? .ready : .notInstalled)
                }
            }
        } catch {
            localEngineProblems[id] = error.localizedDescription
            modelRows[id]?.needsCodeConsent = false; modelRows[id]?.verification = error.localizedDescription
            let dtype = modelRows[id]?.selectedPrecision ?? m.defaultPrecision
            modelRows[id]?.states[dtype] = .init(state: .failed, error: error.localizedDescription, phase: "localEngineError")
        }
    }
    public func approveLocalCode(id: String, contentDigest: String) async throws { try await approveLocalCode(id: id, contentDigest: contentDigest, approvedBy: .cli) }
    public func approveLocalCode(id: String, contentDigest: String, approvedBy: ApprovalSource) async throws {
        try admitWork(); try await discoverLocalModels()
        try await requireApprovableFolder(id) // a broken folder says why; its pins are checked against Hugging Face
        guard brokenLocalFolders[id] == nil, let local = localModels[id], local.contentDigest == contentDigest else {
            throw WorkerFailure(code: "code_consent", message: "The local folder changed; review its new contents before approving.")
        }
        if let runner = selectedRunner(local.manifest) { try artifacts.validateEngine(model: local.manifest, runner: runner) }
        // The owner's yes to these exact contents is enough for a local drop-in, also after its runner code changed: no parity or
        // speed evidence is asked for. requireSwapEvidence now
        // only gates a swap for an older configuration that holds a consent without its approval record.
        let old = config.localCodeConsent[id], oldApproval = config.localCodeApprovals[id], oldRejected = config.rejectedRevisions[id]
        config.rejectedRevisions[id] = nil
        config.localCodeConsent[id] = contentDigest; config.localCodeApprovals[id] = CodeApproval(by: approvedBy, digest: contentDigest)
        do { try persist() } catch { config.localCodeConsent[id] = old; config.localCodeApprovals[id] = oldApproval; config.rejectedRevisions[id] = oldRejected; throw error }
        modelRows[id]?.approval = config.localCodeApprovals[id]; rejectedRevisionDigests[id] = nil
        modelRows[id]?.needsCodeConsent = false
        if residents[id] == nil, let previous = dormantRevisions[id], previous.digest != contentDigest { previousRevisions[id] = previous }
        updatePending(id); updateRevisionRow(id); scheduleRevisionSwap(id); changed()
    }
    public func approveLocalCode(id: String) async throws {
        try await discoverLocalModels()
        guard let local = localModels[id] else { throw WorkerFailure(code: "not_found", message: "Unknown local model \(id).") }
        try await approveLocalCode(id: id, contentDigest: local.contentDigest)
    }
    func checkLocalCode(_ id: String, allowBrokenResident: Bool = false, notifyChanges: Bool = true) async throws {
        if let localCodeCheckBarrier { await localCodeCheckBarrier() }
        guard localModels[id] != nil || brokenLocalFolders[id] != nil else { return }
        try await discoverLocalModels(notifyChanges: notifyChanges)
        if let problem = brokenLocalFolders[id] {
            if allowBrokenResident && residents[id] != nil { return }
            throw WorkerFailure(code: "local_folder", message: problem)
        }
        guard let local = localModels[id] else { throw WorkerFailure(code: "not_found", message: "Local model \(id) was removed.") }
        if let runner = selectedRunner(local.manifest) { try artifacts.validateEngine(model: local.manifest, runner: runner) }
        if config.rejectedRevisions[id] == local.contentDigest, residents[id] == nil, dormantRevisions[id] == nil {
            throw WorkerFailure(code: "revision_rejected", message: "This revision was rolled back or failed to load. Review and approve a repaired revision before Load.")
        }
        if try Self.localCodeRequiresConsent(local), config.localCodeConsent[id] != local.contentDigest {
            throw WorkerFailure(
                code: "code_consent", message: "This local model runs unreviewed code on your Mac; approve it before Get or Load.",
                details: ["id": .string(id), "contentDigest": .string(local.contentDigest)])
        }
    }
    func executableFolder(id: String, installing: Bool) async throws -> URL {
        if let local = localModels[id] {
            try await checkLocalCode(id)
            guard let current = localModels[id], current.contentDigest == local.contentDigest else {
                throw WorkerFailure(code: "code_consent", message: "The local folder changed; approve it again.")
            }
            let support = paths.support
            let protected = protectedRevisionDigests()
            return try await Task.detached {
                let folder = try LocalCodeSnapshot.install(current, support: support)
                try LocalCodeSnapshot.prune(support: support, protecting: protected)
                return folder
            }.value
        }
        if installing { return try await installer.install(id: id, source: source, replacingExisting: true) }
        return try await installer.verifiedOrRefresh(id: id, source: source)
    }
    static let chip = HostChip.current
    func eligibleRunners(_ model: ModelManifest) -> [RunnerSpec] {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        return model.runners.filter { r in
            let min = r.minMacOS.split(separator: ".").compactMap { Int($0) }
            let current = [os.majorVersion, os.minorVersion, os.patchVersion]
            let versionOK = !current.lexicographicallyPrecedes(min + Array(repeating: 0, count: max(0, 3 - min.count)))
            return r.proto == 1 && versionOK && (r.chips.contains("*") || r.chips.contains(Self.chip) || r.chips.contains(RunnerSelection.family(Self.chip)))
        }
    }
    func defaultRunner(_ model: ModelManifest) -> String? { model.selection?.runner(chip: Self.chip) ?? eligibleRunners(model).first?.name }
    func selectedRunner(_ model: ModelManifest) -> RunnerSpec? {
        let eligible = eligibleRunners(model)
        if let override = config.runnerOverrides[model.id], let runner = eligible.first(where: { $0.name == override }) { return runner }
        return eligible.first { $0.name == defaultRunner(model) }
    }
    private func updateRunnerRow(_ model: ModelManifest) {
        modelRows[model.id]?.tools = model.resolvedTools
        modelRows[model.id]?.activity = model.activitySection
        modelRows[model.id]?.toolsShown = config.selections[model.id]?.toolsShown ?? true
        let runner = selectedRunner(model), id = model.id
        modelRows[id]?.precisionDtypes = model.precisionDtypes
        modelRows[id]?.precisionNotes = model.precisionNotes
        modelRows[id]?.formats = model.formats
        modelRows[id]?.runner = runner?.name ?? "Unavailable"
        modelRows[id]?.runnerLanguage = runner?.language ?? ""
        modelRows[id]?.fastExact = runner?.fastExact ?? false
        modelRows[id]?.runnerOverride = config.runnerOverrides[id].flatMap { name in runner?.name == name ? name : nil }
        modelRows[id]?.defaultRunner = defaultRunner(model)
        modelRows[id]?.residentRunner = loadedRunners[id]
        let effectiveFast = modelRows[id]?.fastAvailable ?? false
        modelRows[id]?.fast = effectiveFast
    }
    public func setRunner(id: String, runner: String?) async throws {
        try admitWork(); await refreshLocalModels()
        let model = try manifest(id)
        guard gets[id] == nil && loadTasks[id] == nil else { throw WorkerFailure(code: "busy", message: "Wait for Get or Load to finish before changing the runner.") }
        let name = runner ?? defaultRunner(model)
        guard let choice = eligibleRunners(model).first(where: { $0.name == name }) else {
            throw WorkerFailure(code: "invalid_request", message: "That runner does not support this Mac.")
        }
        try artifacts.validateEngine(model: model, runner: choice)
        let previous = config.runnerOverrides[id]; config.runnerOverrides[id] = runner == defaultRunner(model) ? nil : runner
        do { try persist() } catch { config.runnerOverrides[id] = previous; throw error }
        if let local = localModels[id] { refreshLocalEngine(local) }
        updateRunnerRow(model); try enqueueMeasurements(model); updateMetrics(model); updatePending(id); await refreshInstalled(forceDownloadSizes: true); changed()
    }
    private func cacheEvidence(_ model: ModelManifest, files: [String: Data], local: LocalModelFolder? = nil) {
        resultCache[model.id] = files.keys.sorted().filter { $0.hasPrefix("results/") }.compactMap { path in
            guard let data = files[path], let result = try? JSONDecoder().decode(ResultsFile.self, from: data), result.model == model.id, result.kind == model.kind else {
                return nil
            }; return result
        }
        cachePrecisionEvidence(model, files: files, local: local)
        cacheFolderFigureRecords(model, files: files, local: local)
        runnerResultsCache[model.id] = Dictionary(
            grouping: (resultCache[model.id] ?? []).filter { RunnerSelection.family($0.chip) == RunnerSelection.family(Self.chip) }, by: \.runner
        ).mapValues { values in values.compactMap { try? WireJSON.decode(JSONEncoder().encode($0)) } }
        historyCache[model.id] = Dictionary(
            uniqueKeysWithValues: model.runners.compactMap { runner in CatalogHistory.latest(files, runner: runner.name).map { (runner.name, $0) } })
    }
    private func cachedRunner(_ model: ModelManifest, precision: String, fast: Bool, folder: URL) async throws -> RunnerSpec {
        guard let runner = selectedRunner(model) else { throw WorkerFailure(code: "runner", message: "The selected runner does not support this Mac.") }
        _ = try await artifacts.launch(model: model, runner: runner, directory: folder, paths: paths)
        modelRows[model.id]?.runner = runner.name; modelRows[model.id]?.runnerLanguage = runner.language; modelRows[model.id]?.fastExact = runner.fastExact
        return runner
    }
    func updateMetrics(_ model: ModelManifest) {
        guard let row = modelRows[model.id] else { return }
        let precisionStats = statsForRunner(model, runner: row.runner)
        modelRows[model.id]?.precisionStats = precisionStats
        let memoryNote = loadMemoryRequirement(model, precision: row.selectedPrecision).neededGB == nil ? LoadMemoryRequirement.unknownNote : nil
        modelRows[model.id]?.loadMemoryNote = memoryNote
        modelRows[model.id]?.recommendedPrecision = nil
        modelRows[model.id]?.recommendationPending = false
        modelRows[model.id]?.recommendationReason = ""
        let results = currentResults(model).filter {
            $0.runner == row.runner && $0.model == model.id && $0.kind == model.kind && $0.precision == row.selectedPrecision && $0.mode == (row.fast ? "fast" : "exact")
        }
        let badge = MeasurementBadge.derive(results: results, chip: Self.chip)
        let shown =
            results.first { $0.chip == Self.chip && $0.eligibleMeasurement } ?? results.first { $0.chip == Self.chip && $0.verificationLabel == "quick check" }
            ?? results.first { $0.eligibleMeasurement }
        var metrics = shown?.tableMetrics(columns: kinds().first { $0.id == model.kind }?.columns).mapValues { JSONValue.string($0) } ?? [:]
        updateContextMetrics(model, row: row, metrics: &metrics)
        modelRows[model.id]?.metrics = metrics
        modelRows[model.id]?.badge = ModelBadge(rawValue: badge.rawValue) ?? .unverified
        modelRows[model.id]?.verification =
            brokenLocalFolders[model.id] ?? localEngineProblems[model.id]
            ?? (row.local
                ? "Local · unsigned folder" : shown?.verificationLabel == "quick check" ? "quick check" : badge == .unverified ? "Not tried on this Mac yet" : badge.rawValue)
        // Only the run's own figures are rough; context comes from the manifest.
        modelRows[model.id]?.quickMetrics = shown?.quickCheck == true ? metrics.filter { !["context", "native_context"].contains($0.key) }.mapValues { _ in true } : [:]
        modelRows[model.id]?.peakRAMMB = shown?.metrics["peak_ram_mb"] ?? nil
        modelRows[model.id]?.folderFigures = shownFolderFigures(model, row: row)
        updateChecks(model); updateRevisionRow(model.id)
    }
    func manifest(_ id: String) throws -> ModelManifest {
        guard let m = catalog.models.first(where: { $0.id == id })?.manifest ?? localModels[id]?.manifest ?? droppedCatalogModels[id]?.manifest ?? residentManifests[id] else {
            throw WorkerFailure(code: "not_found", message: "Unknown model \(id).")
        }; return m
    }
    private func revision(_ m: ModelManifest, _ precision: String) -> String { m.vendor?[precision]?.revision ?? m.weights.revision }
    func snapshot(_ m: ModelManifest, _ precision: String) -> URL {
        paths.directory("weights").appendingPathComponent("snapshots/\(m.id)/\(revision(m, precision))/\(precision)")
    }
    func snapshotPresent(_ m: ModelManifest, _ precision: String) -> Bool {
        installationUpdates[m.id]?.contains(precision) != true && FileManager.default.fileExists(atPath: StoredNames.snapshotURL(in: snapshot(m, precision)).path)
    }
    func downloadSources(_ m: ModelManifest, precision: String) -> [WeightSpec] {
        var sources: [WeightSpec] = []
        if !snapshotPresent(m, precision) {
            if let vendor = m.vendor?[precision] {
                sources.append(vendor)
            } else if precision == m.basePrecision || !snapshotPresent(m, m.basePrecision) {
                sources.append(m.weights)
            }
        }
        // Incorporated assets are downloaded only while the base snapshot is still to be converted.
        return sources + (snapshotPresent(m, m.basePrecision) ? m.separateAssets : m.selectedAssets).map(\.weights)
    }
    private func makeDownloadProgress(_ model: ModelManifest, precision: String) async throws -> GetProgressAccumulator {
        var missing: [PendingDownloadFile] = []
        for source in downloadSources(model, precision: precision) {
            for file in source.files {
                guard let hash = source.sha256?[file], let bytes = source.bytes?[file] else {
                    throw WorkerFailure(code: "download_size", message: "Get needs pinned hashes and file sizes.")
                }
                if try await !fetcher.store.isVerifiedPresent(sha256: hash, bytes: bytes) { missing.append(PendingDownloadFile(contentID: hash, file: file, bytes: bytes)) }
            }
        }
        return try GetProgressAccumulator(files: missing)
    }
    /// Why a declared runner cannot run here, in the order the eligibility check applies (protocol, macOS, chip).
    func ineligibleReason(_ runner: RunnerSpec) -> String {
        if runner.proto != 1 { return "Needs a newer Arsenal" }
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let min = runner.minMacOS.split(separator: ".").compactMap { Int($0) }
        if [os.majorVersion, os.minorVersion, os.patchVersion].lexicographicallyPrecedes(min + Array(repeating: 0, count: max(0, 3 - min.count))) {
            return "Needs macOS \(runner.minMacOS)"
        }
        let chips = runner.chips.map { $0.uppercased() }.joined(separator: ", ")
        return chips.isEmpty ? "Not available on this Mac" : "Built for \(chips)"
    }
    private func availability(_ model: ModelManifest, precision: String) async throws -> GetAvailabilityPlan {
        var planned = model
        if !getReplans.contains(model.id + ":" + precision),
            snapshotPresent(model, precision) || (model.vendor?[precision] == nil && precision != model.basePrecision && snapshotPresent(model, model.basePrecision))
        {
            if var vendor = planned.vendor?[precision] { vendor.files = []; planned.vendor?[precision] = vendor } else { planned.weights.files = [] }
        }
        return try await fetcher.availabilityPlan(model: planned, precision: precision)
    }
    func refreshInstalled(forceDownloadSizes: Bool = false, notifyChanges: Bool = true) async {
        await refreshExitedResidents(notifyChanges: notifyChanges) { [supervisor] in await supervisor.hasExited(id: $0) }
        for m in allManifests {
            guard let row = modelRows[m.id] else { continue }
            let snapshotFiles = m.precisions.map { InstalledRowCacheKey.File(StoredNames.snapshotURL(in: snapshot(m, $0))) }
            let key = InstalledRowCacheKey(
                snapshots: snapshotFiles,
                nativeConfig: m.kind == "chat" ? .init(snapshot(m, row.selectedPrecision).appendingPathComponent("config.json")) : nil,
                localResults: InstalledRowCacheKey.localResults(folder: paths.directory("models").appendingPathComponent(m.id)),
                retention: .init(paths.directory("weights").appendingPathComponent("retention/\(m.id)/\(m.weights.revision)/\(m.basePrecision).json")),
                dtype: row.selectedPrecision, fast: row.fast, runner: row.runner,
                localDigest: row.contentDigest, problem: brokenLocalFolders[m.id] ?? localEngineProblems[m.id])
            for (p, snapshotFile) in zip(m.precisions, snapshotFiles) {
                if forceDownloadSizes, let plan = try? await availability(m, precision: p) {
                    modelRows[m.id]?.downloadBytes[p] = plan.networkBytes; modelRows[m.id]?.localCopyBytes[p] = plan.localCopyBytes
                }
                let current = modelRows[m.id]?.states[p]?.state
                if installationUpdates[m.id]?.contains(p) == true
                    && (current == nil || current == .ready || current == .notInstalled || modelRows[m.id]?.states[p]?.phase == "updateAvailable")
                {
                    modelRows[m.id]?.states[p] = .init(state: .updateAvailable, phase: "updateAvailable")
                    continue
                }
                if current == nil || current == .ready || current == .notInstalled {
                    var present = snapshotFile.exists && installationUpdates[m.id]?.contains(p) != true
                    // Ready means loadable: a snapshot whose files no longer match their pins is not ready, though its bytes stay listed.
                    if present { present = await snapshotIntact(m, p) }
                    let state: ModelState = present ? .ready : .notInstalled
                    if modelRows[m.id]?.states[p]?.state != state { modelRows[m.id]?.states[p] = PrecisionState(state: state) }
                }
            }
            guard forceDownloadSizes || installedRowCache[m.id] != key else { continue }
            if let receipt = try? await fetcher.store.sourceRetentionReceipt(model: m.id, revision: m.weights.revision, sourcePrecision: m.basePrecision) {
                let note = SourceRetentionPolicy.rowNote(receipt, downloadBytes: modelRows[m.id]?.downloadBytes[m.basePrecision])
                modelRows[m.id]?.retentionNote = note
                if modelRows[m.id]?.states[m.basePrecision]?.state == .notInstalled { modelRows[m.id]?.states[m.basePrecision]?.phase = PrecisionState.sourceRemovedPhase }
            } else {
                modelRows[m.id]?.retentionNote = nil
                if modelRows[m.id]?.states[m.basePrecision]?.phase == PrecisionState.sourceRemovedPhase { modelRows[m.id]?.states[m.basePrecision]?.phase = nil }
            }
            // Tiers the entry no longer offers stay on disk until the user removes them (`deleteWithdrawnTiers`): reported, never
            // silently kept or silently deleted. Await before the optional-chained assignment: awaiting inside it holds
            // the row moved out of the dictionary across the suspension, and calls interleaving there free it twice (2026-10-22 crash).
            let withdrawn = await fetcher.store.withdrawnPrecisions(model: m.id, offered: Set(m.precisions))
            modelRows[m.id]?.withdrawnTiers = withdrawn
            updateMetrics(m)
            installedRowCache[m.id] = key
        }
    }
    public func rows() async -> [ModelRow] {
        await refreshLocalModels(notifyChanges: false); await refreshInstalled(notifyChanges: false)
        // A worker running a model off the table (an agent's, or one whose membership a file edit removed) stays visible
        // to status and Unload, marked so the Models table, menu and tool windows leave it out.
        // The table is the models with weights on this Mac or a Get under way, in the order they joined it.
        let present = Set(modelRows.keys.filter(onTable))
        let order = config.table.filter(present.contains) + present.filter { !config.table.contains($0) }.sorted()
        let offTable = residents.keys.filter { !present.contains($0) }.sorted()
        return (order + offTable).compactMap { id in
            modelRows[id].map { row in
                var row = row; row.onTable = present.contains(id); return row
            }
        }
    }
    public func catalog(query: CatalogQuery = .init()) async -> [CatalogEntry] {
        await refreshLocalModels(notifyChanges: false); await refreshInstalled(notifyChanges: false)
        var entries = allManifests.compactMap { m -> CatalogEntry? in
            guard let row = modelRows[m.id] else { return nil }
            let key = CatalogEntryCacheKey(row: row, added: onTable(m.id), settings: modelSettings.values[m.id])
            if let cached = catalogEntryCache[m.id], cached.key == key { return cached.entry }
            let sizes = m.sizesByDtype
            var entry = CatalogEntry(row: row, metadata: metadata[m.id] ?? [:], sizesByDtype: sizes, added: key.added)
            entry.formats = m.formats
            for f in m.formats {
                // The figure Load admission uses (the table's recorded peak, then the folder's forecast) comes first: the
                // weight size alone understated SeedVR2 at 9.2 GB against a 38.7 GB recorded peak (core review, 2 Nov).
                entry.ramMBByDtype[f.id] = CatalogEntry.ramMB(
                    row: row, precision: f.id, forecast: m.memory, measuredMB: memoryMeasurement(m, precision: f.id, fast: row.fast), declaredMB: f.ramMB,
                    sizeBytes: sizes[f.id])
            }
            entry.settleRAM(selected: row.selectedPrecision, physicalBytes: ProcessInfo.processInfo.physicalMemory)
            entry.runners = runnerDetails(m); entry.settingsSummary = modelSettings.states(m.id, loaded: residents[m.id] != nil)
            catalogEntryCache[m.id] = (key, entry)
            return entry
        }
        entries += droppedCatalogEntries
        entries += localProblemIDs.sorted().compactMap { modelRows[$0].map { CatalogEntry(row: $0, added: false) } }
        entries += brokenLocalFolders.keys.sorted().filter { localModels[$0] == nil }.compactMap { id in
            modelRows[id].map { CatalogEntry(row: $0, added: onTable(id)) }
        }
        return CatalogEntry.query(entries, query)
    }
    func recommendedDtype(_ model: ModelManifest) -> String {
        let runner = selectedRunner(model)?.name ?? "Unavailable"
        return servablePrecision(model, PrecisionRecommendation.choose(statsForRunner(model, runner: runner), baseDtype: model.basePrecision, chip: Self.chip).dtype)
    }

    public func remove(id: String, deleteWeights: Bool = false) async throws {
        try admitWork(); guard modelRows[id] != nil else { throw WorkerFailure(code: "not_found", message: "Unknown model \(id).") }
        if let swap = swapTasks[id] { swap.cancel(); await swap.value }
        try await waitForRevisionSwap(id)
        // A refused Remove must not interrupt an independent Get of another dtype.
        try requireIdleResident(id, forDelete: deleteWeights)
        await cancelGet(id: id)
        if deleteWeights { try await delete(id: id) } else { try await unload(id: id) }
        // A failed Get left no weights: Remove takes its row off the table.
        for (p, s) in modelRows[id]?.states ?? [:] where s.state == .failed && [nil, "get_plan_changed"].contains(s.phase) { resetInstalledState(id, precision: p) }
        let previous = config.table; config.table.removeAll { $0 == id }
        do { try persist() } catch { config.table = previous; throw error }; changed()
        if droppedCatalogModels.removeValue(forKey: id) != nil { modelRows[id] = nil; metadata[id] = nil }
        if brokenLocalFolders[id] != nil {
            localModels[id] = nil; metadata[id] = nil; operationCache[id] = nil; operationFailures[id] = nil; modelSettings.declarations[id] = nil
            modelRows[id] = nil; brokenLocalFolders[id] = nil
            localEngineProblems[id] = nil; installedRowCache[id] = nil; catalogEntryCache[id] = nil
        }
    }
    public func settings(id: String) async throws -> [SettingState] {
        _ = try manifest(id)
        var states = modelSettings.states(id, loaded: residents[id] != nil)
        let voiceSettings = states.indices.filter { states[$0].declaration["type"]?.string == "voice" }
        if !voiceSettings.isEmpty {
            // Metadata only: no runner request, audio read, load or service event from this getter.
            let inventory = try SavedVoiceStore.list(model: id, support: paths.support)
            let fingerprint = FileDigest.sha256(try WireJSON.encode(["voices": .array(inventory)]))
            for index in voiceSettings { states[index].declaration["saved_voices_sha256"] = .string(fingerprint) }
        }
        return states
    }
    public func setSetting(id: String, key: String, value: JSONValue) async throws { try await setSettings(id: id, values: [key: value]) }
    public func saveDesignedVoice(id: String, previewID: String, name: String) async throws -> JSONObject {
        try admitWork(); let ops = try await resolvedOperations(id: id)
        guard ops.contains(where: { $0.name == "design_voice" }) else { throw WorkerFailure(code: "unsupported_op", message: "This model does not offer voice design.") }
        let support = paths.support
        let saved = try await Task.detached { try DesignedVoicePreviews.save(model: id, previewID: previewID, name: name, support: support) }.value
        changed(); return saved
    }
    public func deleteSavedVoice(id: String, voiceID: String) async throws {
        try admitWork(); _ = try manifest(id)
        let support = paths.support
        try await Task.detached { try SavedVoiceStore.delete(model: id, voiceID: voiceID, support: support) }.value
        let reset = (modelSettings.declarations[id] ?? []).compactMap { declaration -> String? in
            guard declaration["type"]?.string == "voice", let key = declaration["id"]?.string, modelSettings.values[id]?[key] == .string(voiceID) else { return nil }
            return key
        }
        if !reset.isEmpty {
            for key in reset { modelSettings.remove(id, key: key) }
            try persist()
            settingsWarnings[id] = "Saved voice was deleted; using the model default."
            updatePending(id)
        }
        changed()
    }
    func updatePending(_ id: String) {
        modelRows[id]?.settingsWarning = modelSettings.warning(id) ?? settingsWarnings[id]
        let row = modelRows[id], loaded = loadedPreferences[id]
        let selectable = row.map { [.ready, .loaded].contains($0.states[$0.selectedPrecision]?.state ?? .notInstalled) } ?? false
        let selectionPending =
            loaded != nil
            && (loaded?.precision != row?.selectedPrecision
                || (selectable
                    && (loaded?.fast != row?.fast || loadedRunners[id] != row?.runner
                        || !sameImplementation(id))))
        modelRows[id]?.needsReload =
            (residents[id] != nil && residentManifests[id] != (try? manifest(id))) || (residents[id] != nil && loadedRunners[id] != row?.runner) || selectionPending
            || modelSettings.states(id, loaded: residents[id] != nil).contains { $0.pending }
    }
    public func applySettings(id: String) async throws {
        try admitWork(); _ = try manifest(id)
        try await checkLocalCode(id)
        try await ensureLoaded(id: id, precision: nil, fast: nil, residency: .manual, forceReload: true)
    }
    public func events() -> AsyncStream<ServiceEvent> {
        let id = UUID(); let (s, c) = AsyncStream<ServiceEvent>.makeStream(bufferingPolicy: .bufferingNewest(64)); listeners[id] = c
        c.onTermination = { [weak self] _ in Task { await self?.removeListener(id) } }; c.yield(.readFeedback(.catalog(catalogStatus()))); return s
    }
    private func removeListener(_ id: UUID) { listeners[id] = nil }
    func emit(_ e: ServiceEvent) { for c in listeners.values { c.yield(e) } }
    func changed() { activityAggregator.changed(); emit(.rowsChanged) }
    func admitNewWork() throws {
        guard !updating else { throw WorkerFailure(code: "updating", message: "Arsenal is preparing an update; retry after it relaunches.") }
    }
    func admitWork(allowRefresh: Bool = false) throws {
        guard !shuttingDown else { throw WorkerFailure(code: "shutting_down", message: "Arsenal is shutting down; start it again before using a model.") }
        try Task.checkCancellation(); guard !refreshing || allowRefresh else { throw WorkerFailure(code: "busy", message: "Refresh is in progress; try again when it finishes.") }
    }
}

extension RealModelsService {
    public func get(id: String, precision: String) async throws {
        let plan = try await getPlan(id: id, dtype: precision)
        try await get(id: id, precision: precision, approvedPlan: plan)
    }
    public func get(id: String, precision: String, approvedPlan: GetPlan) async throws {
        try admitNewWork(); try admitWork()
        let current = try await getPlan(id: id, dtype: precision)
        try current.requireKnownDownloadSize()
        guard approvedPlan.covers(current) else { throw GetPlanChanged(revisedPlan: current) }
        if let task = gets[id] {
            try await task.value; while gets[id] != nil { await Task.yield() }; if let m = try? manifest(id), snapshotPresent(m, precision) { return }
            try await get(id: id, precision: precision, approvedPlan: approvedPlan); return
        }
        try await checkLocalCode(id)
        let m = try manifest(id); guard m.precisions.contains(precision) else { throw WorkerFailure(code: "precision", message: "That dtype is not available.") }
        try await requirePinnedFiles(id, precision: precision) // what this precision downloads, also without code approval
        guard let runner = selectedRunner(m), m.supports(dtype: precision, runner: runner.name) else {
            throw WorkerFailure(code: "unsupported_format", message: modelRows[id]?.dtypeUnavailableReason(precision) ?? "No runner in this Arsenal yet.")
        }
        if let reason = modelRows[id]?.dtypeUnavailableReason(precision) { throw WorkerFailure(code: "unsupported_format", message: reason) }
        try await joinTable(id, precision: precision)
        // Joining the table can wait (a local-folder scan): a Get of this model that started meanwhile is joined instead.
        if gets[id] != nil { try await get(id: id, precision: precision, approvedPlan: approvedPlan); return }
        try admitNewWork() // No suspension between the gate check and registering admitted work.
        // An automatic retry whose wait was cancelled or removed meanwhile (`GateRetry.episode`) starts nothing.
        if let episode = GateRetry.episode, !gateRetry.isCurrent(id, episode: episode) { throw CancellationError() }
        // A Get started while a gated one waits (a click): its counter goes; the wait goes on until this one ends.
        // After every refusal above, so a refused click never leaves the ring stopped at 0.
        gateRetry.retryingNow(id)
        let task = Task { try await self.performGet(m, precision: precision, approvedPlan: approvedPlan) }; gets[id] = task; getPrecisions[id] = precision
        changed() // Registered: Refresh is refused from here (`refreshRefusal`), also while this Get waits for a running Load.
        do {
            try await task.value
            gets[id] = nil; getPrecisions[id] = nil; getReplans.remove(id + ":" + precision)
            if getReplanPrecision[id] == precision { getReplanPrecision[id] = nil }
            gateRetry.end(id); clearGateRows(id, except: precision)
            var state = modelRows[id]?.states[precision] ?? PrecisionState()
            state.state = loadedPreferences[id]?.precision == precision ? .loaded : .ready
            state.phase = nil; state.error = nil; state.gate = nil; modelRows[id]?.states[precision] = state
            updatePending(id); scheduleRevisionSwap(id); changed()
        } catch {
            gets[id] = nil; getPrecisions[id] = nil
            if error is CancellationError {
                gateRetry.end(id)
                resetInstalledState(id, precision: precision); changed(); throw error
            }
            if let gate = error as? HFGateError {
                guard loadedPreferences[id]?.precision == precision else { gated(id, precision: precision, gate: gate); throw gate }
                // The loaded precision keeps running, so there is no wait to count down: the page opens and the refusal
                // is said like any failed Get (the row's error, an error event).
                openPages(gate.pages)
            }
            gateRetry.end(id); clearGateRows(id, except: precision)
            var failure: any Error = error
            if (error as? WorkerFailure)?.code == "get_plan_changed" {
                getReplans.insert(id + ":" + precision); getReplanPrecision[id] = precision
                do { failure = GetPlanChanged(revisedPlan: try await getPlan(id: id, dtype: precision)) } catch { failure = error }
            }
            modelRows[id]?.states[precision] = .init(
                state: loadedPreferences[id]?.precision == precision ? .loaded : .failed, error: failure.localizedDescription,
                phase: failure is GetPlanChanged ? "get_plan_changed" : Self.pinnedFileMissingPhase(failure))
            emit(.error(id: id, message: error.localizedDescription)); changed(); throw failure
        }
    }
    private func performGet(_ m: ModelManifest, precision: String, approvedPlan: GetPlan) async throws {
        try admitWork()
        let id = m.id
        defer { getProgress[id] = nil }
        if let loading = loadTasks[id] { try await loading.value; try Task.checkCancellation() }
        modelRows[id]?.selectedPrecision = precision
        modelRows[id]?.states[precision] = .init(state: .downloading, progress: 0); updatePending(id); changed()
        let folder = try await executableFolder(id: id, installing: true)
        guard let runner = selectedRunner(m) else { throw WorkerFailure(code: "runner", message: "No runner supports this Mac.") }
        try await artifacts.prepare(model: m, runner: runner, directory: folder, approvedPlan: approvedPlan)
        let progress = try await makeDownloadProgress(m, precision: precision); getProgress[id] = progress
        modelRows[id]?.states[precision] = .init(state: .downloading, progress: 0, bytes: 0, totalBytes: progress.totalBytes, bytesPerSecond: 0); changed()
        let verifiedKinds = try await installer.verifiedKindsDirectory()
        let delivery = GetProgressDelivery { [weak self] values, final in await self?.progress(id: id, precision: precision, values: values, publish: !final) }
        do {
            _ = try await fetcher.get(
                model: m, modelDirectory: folder, precision: precision, kindsDirectory: verifiedKinds,
                converter: WorkerPrecisionConverter(artifacts: artifacts, paths: paths, runnerNames: [id: runner.name]),
                approvedDownloads: approvedPlan.items.first { $0.id == "weights" }?.networkContent ?? [:], progress: { p in delivery.submit(p) },
                deriving: { delivery.submitDeriving() })
            await delivery.finish()
        } catch { await delivery.finish(discard: true); throw error }
        // Get can also repair the base precision/assets while rebuilding a derived precision.
        for p in m.precisions {
            if try await fetcher.updateRequired(model: m, precision: p) {
                installationUpdates[id, default: []].insert(p)
            } else {
                installationUpdates[id]?.remove(p)
                if modelRows[id]?.states[p]?.phase == "updateAvailable" { modelRows[id]?.states[p] = nil }
            }
        }
        _ = try await cachedRunner(m, precision: precision, fast: modelRows[id]?.fast ?? false, folder: folder)
        try Task.checkCancellation(); getProgress[id]?.finish()
        modelRows[id]?.states[precision] = .init(
            state: .downloading, progress: 1, bytes: getProgress[id]?.doneBytes, totalBytes: getProgress[id]?.totalBytes, bytesPerSecond: getProgress[id]?.rate(),
            phase: "verifying")
        modelRows[id]?.selectedPrecision = precision
        var preference = config.selections[id] ?? .init(precision: precision)
        preference.precision = precision; preference.explicitSelection = true; preference.keepHot = modelRows[id]?.keepHot ?? false
        config.selections[id] = preference
        persistSelectionBestEffort(id)
        await refreshInstalled(forceDownloadSizes: true); changed()
    }
    /// Download counts and conversion markers arrive in order. A Get that converts one source shard and then
    /// downloads the next shows downloading again, never a frozen "deriving" while bytes arrive.
    private func progress(id: String, precision: String, values: [DownloadProgress], publish: Bool) {
        guard gets[id] != nil, let before = modelRows[id]?.states[precision]?.state, [.downloading, .deriving].contains(before), var progress = getProgress[id]
        else { return }
        var latest: PrecisionState?
        for value in values { if let state = progress.record(value) { latest = state } }
        getProgress[id] = progress
        guard let state = latest else { return }
        modelRows[id]?.states[precision] = state
        if state.state != before { changed() }
        if publish { emit(.progress(id: id, precision: precision, progress: progress.fraction)) }
    }
    public func cancelGet(id: String) async {
        endGateWait(id)
        guard let task = gets[id] else {
            if let precision = getReplanPrecision.removeValue(forKey: id) { resetInstalledState(id, precision: precision) }
            changed(); return
        }
        let precision = getPrecisions[id]
        task.cancel(); _ = try? await task.value
        // The owning Get finishes its row/dictionary cleanup after the task completes.
        while gets[id] != nil { await Task.yield() }
        if let precision { resetInstalledState(id, precision: precision) }
        changed()
    }
    func resetInstalledState(_ id: String, precision: String) {
        let installed = (try? manifest(id)).map { snapshotPresent($0, precision) } ?? false
        let sourceRemoved = !installed && precision == (try? manifest(id))?.basePrecision && modelRows[id]?.retentionNote != nil
        modelRows[id]?.states[precision] = .init(
            state: loadedPreferences[id]?.precision == precision
                ? .loaded : installationUpdates[id]?.contains(precision) == true ? .updateAvailable : installed ? .ready : .notInstalled,
            phase: installationUpdates[id]?.contains(precision) == true && loadedPreferences[id]?.precision != precision
                ? "updateAvailable" : sourceRemoved ? PrecisionState.sourceRemovedPhase : nil)
    }
    public func load(id: String, precision: String?, fast: Bool?) async throws {
        while reloading.contains(id) { try await waitForMemoryChange(); try Task.checkCancellation() }
        try admitNewWork(); try await ensureLoaded(id: id, precision: precision, fast: fast, residency: .manual)
        // Only an explicit dtype choice changes the saved precision. Reloads and automatic
        // revision swaps describe the resident; their captured dtype is never a preference.
        if precision != nil || fast != nil {
            let model = try manifest(id), selected = modelRows[id]?.selectedPrecision ?? model.defaultPrecision
            var preference = config.selections[id] ?? .init(precision: selected)
            if let precision { preference.precision = precision; preference.explicitSelection = true }
            if let fast, modelRows[id]?.fastAvailable == true { preference.fast = fast }
            config.selections[id] = preference; persistSelectionBestEffort(id); updateMetrics(model); updatePending(id); changed()
            if let weights = try await fetcher.store.verifyInstalledSnapshot(model: model, precision: loadedPreferences[id]?.precision ?? selected) {
                await applySourceRetention(model, loaded: weights)
            }
        }
    }
    func requireCurrentInstallation(_ id: String, precision: String) throws {
        if installationUpdates[id]?.contains(precision) == true { throw WorkerFailure(code: "update_available", message: updateMessage(id, precision)) }
    }
    func ensureLoaded(id: String, precision: String?, fast: Bool?, residency: ResidencyClass, forceReload: Bool = false, waitForMemory: Bool = true) async throws {
        try admitNewWork()
        try requireCurrentCatalogModel(id)
        guard loadable(id) else { throw Self.notOnTable(id) }
        try admitWork()
        if try await restoreDormantRevisionIfNeeded(id, residency: residency) { return }
        let (m, p, _) = try await loadSelection(id: id, precision: precision)
        if let sharedLoad = loadTasks[id] {
            // Joining an existing load allocates nothing. A different requested precision is checked
            // after the shared task completes, when ensureLoaded determines whether another load is needed.
            while loadTasks[id] != nil { try await waitForMemoryChange() }
            try Task.checkCancellation()
            do { try await sharedLoad.value } catch is CancellationError {
                // Cancellation belongs to the owning caller; an independent waiter can start a fresh load.
                try Task.checkCancellation()
            }
            try await ensureLoaded(id: id, precision: p, fast: fast, residency: residency, forceReload: forceReload, waitForMemory: waitForMemory); return
        }
        if !forceReload, residents[id] != nil, loadedPreferences[id]?.precision == p, loadedPreferences[id]?.fast == (modelRows[id]?.fastAvailable == true),
            loadedRunners[id] == selectedRunner(m)?.name, sameImplementation(id), !(await supervisor.hasExited(id: id))
        {
            // Explicit Load rechecks trust even when already resident; ordinary runs use the verified resident.
            if residency == .manual { _ = try await executableFolder(id: id, installing: false); _ = try await fetcher.verifyInstalled(model: m, precision: p) }
            if residency == .manual { residents[id]?.residency = .manual }; touch(id); return
        }
        try requireLoadAssets(m, precision: p)
        let releasing = residents[id] != nil ? !(await supervisor.hasExited(id: id)) : false
        try checkLoadMemory(m, precision: p, releasingResident: releasing)
        let requestedFast = modelRows[id]?.fastAvailable == true
        try admitNewWork() // Trust/supervisor checks above may have suspended across an update handoff.
        let task = Task {
            try await self.performLoad(
                m, precision: p, fast: requestedFast, residency: residency, waitForMemory: waitForMemory)
        }
        // Registered: Refresh is refused from here (`refreshRefusal`), before the row shows Loading.
        loadTasks[id] = task; changed()
        do {
            try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            loadTasks[id] = nil; armTimer(id); signalMemoryChange(); scheduleRevisionSwap(id); changed()
        } catch {
            loadTasks[id] = nil; signalMemoryChange()
            if error is CancellationError {
                resetInstalledState(id, precision: p)
            } else {
                modelRows[id]?.states[p] = .init(state: loadedPreferences[id]?.precision == p ? .loaded : .ready, error: error.localizedDescription)
            }
            changed()
            throw error
        }
    }
    func performLoad(_ m: ModelManifest, precision: String, fast: Bool, residency: ResidencyClass, waitForMemory: Bool) async throws {
        try admitWork()
        let id = m.id
        let expectedDigest = localModels[id]?.contentDigest
        guard let selectedName = selectedRunner(m)?.name, m.supports(dtype: precision, runner: selectedName) else {
            throw WorkerFailure(code: "unsupported_format", message: "The selected runner does not support this dtype.")
        }
        let folder: URL
        do { folder = try await executableFolder(id: id, installing: false) } catch {
            throw WorkerFailure(code: "integrity", message: "\(error.localizedDescription); use \(MCPNames.prefix)download(model: \(id)) to repair.")
        }
        if let expectedDigest, folder.deletingLastPathComponent().lastPathComponent != expectedDigest {
            throw WorkerFailure(code: "revision", message: "The candidate changed during Load; retry its current revision.")
        }
        let installed = try await fetcher.verifyInstalled(model: m, precision: precision)
        guard let weights = try await fetcher.store.verifyInstalledSnapshot(model: m, precision: precision) else {
            throw WorkerFailure(code: "not_installed", message: "Model \(id) is not installed completely; use \(MCPNames.prefix)download(model: \(id)) to repair it.")
        }
        guard let selected = selectedRunner(m) else { throw WorkerFailure(code: "runner", message: "No runner supports this Mac.") }
        try requireNativeRunner(selected)
        // Resolve and validate the native engine before discarding a healthy resident. Load never downloads.
        let runner = selected
        let launch = try await artifacts.launch(model: m, runner: runner, directory: folder, paths: paths)
        let releasing = residents[id] != nil ? !(await supervisor.hasExited(id: id)) : false
        let memoryAdmission = try checkLoadMemory(m, precision: precision, reserve: true, releasingResident: releasing)
        defer { loadReservations[id] = nil; loadReservationReleases.remove(id); signalMemoryChange() }
        let oldRevision = captureRevision(id) ?? dormantRevisions[id]
        if residents[id] != nil { try await unloadResident(id: id) }
        loadReservationReleases.remove(id)
        ActivityRequestContext.tracking?.startLoading()
        if let tracking = ActivityRequestContext.tracking { activityAggregator.phase(tracking.id, "loading") }
        try admitWork()
        modelRows[id]?.states[precision] = .init(state: .loading); changed()
        let ops = try await resolvedOperations(id: id, candidate: true, modelOverride: expectedDigest == nil ? nil : m, folderOverride: expectedDigest == nil ? nil : folder)
        let appliedSettings = modelSettings.loadValues(id)
        let loadConfig = try ModelLoadConfiguration.config(
            model: m, folder: folder, assets: installed.assets, chip: Self.chip, fast: fast, settings: modelSettings,
            maximumContext: m.kind == "chat" ? maximumContext(m, snapshot: installed.weightsDirectory) : nil)
        try admitWork()
        let worker = try await supervisor.worker(
            id: id, launch: launch, manual: residency == .manual, status: { [weak self] memory in await self?.status(id, memory) },
            statusDetails: { [weak self] details in await self?.kernelStatus(id, details) })
        do {
            try await worker.verifyOperations(ops.filter { $0.name != "delete_voice" || m.kind != "voices" }.map(\.name))
            let loadReply = try await worker.request(
                op: "load",
                params: [
                    "model_dir": .string(folder.path), "weights_dir": .string(installed.weightsDirectory.path), "dtype": .string(precision), "fast": .bool(fast),
                    "config": .object(loadConfig)
                ], timeout: 180,
                memoryCeiling: workerMemoryCeiling(id, requirement: memoryAdmission.requirement, precision: precision, forced: LoadMemoryContext.force))
            kernelStatus(id, loadReply)
            modelSettings.applied[id] = appliedSettings
            loadedPreferences[id] = ModelPreferences(precision: precision, fast: fast)
            loadedRunners[id] = runner.name; modelRows[id]?.residentRunner = runner.name
            for dtype in modelRows[id]?.precisions ?? [] where modelRows[id]?.states[dtype]?.phase == "update_restore_failed" { resetInstalledState(id, precision: dtype) }
            await fetcher.store.retain(model: id)
            modelRows[id]?.states[precision] = .init(state: .loaded)
            if localModels[id] != nil { residentLocalDigests[id] = folder.deletingLastPathComponent().lastPathComponent }
            if let oldRevision, oldRevision.digest != residentLocalDigests[id], !reloading.contains(id) { previousRevisions[id] = oldRevision }
            residentMenus[id] = declaredMenuSections(m); residentOperations[id] = ops; residentManifests[id] = m
            residentSettings[id] = modelSettings; residentLaunches[id] = launch
            residentMemoryRequirements[id] = memoryAdmission.requirement
            residentLoads[id] = [
                "model_dir": .string(folder.path), "weights_dir": .string(installed.weightsDirectory.path), "dtype": .string(precision), "fast": .bool(fast),
                "config": .object(loadConfig)
            ]
            residents[id] = ResidentModel(id: id, residency: residency, memoryMB: footprints[id]?.footprintMB ?? 0, keepHot: modelRows[id]?.keepHot ?? false)
            recordResidentMemoryApproval(id)
            updatePending(id); updateMetrics(m)
            await applySourceRetention(m, loaded: weights)
            changed()
        } catch {
            await supervisor.unload(id: id); loadedPreferences[id] = nil; loadedRunners[id] = nil; modelRows[id]?.residentRunner = nil; footprints[id] = nil
            if residents.removeValue(forKey: id) != nil { await fetcher.store.release(model: id) }
            modelRows[id]?.states[precision] = .init(state: .ready)
            if let oldRevision, !reloading.contains(id), !shuttingDown { try? await restoreInterruptedRevision(id, oldRevision) }
            throw error
        }
    }
    private func memoryMeasurement(_ model: ModelManifest, precision: String, fast: Bool) -> Double? {
        currentResults(model).first {
            $0.runner == selectedRunner(model)?.name && $0.precision == precision && $0.mode == (fast ? "fast" : "exact") && $0.chip == Self.chip
                && ($0.eligibleMeasurement || $0.verificationLabel == "quick check")
        }?.metrics["peak_ram_mb"] ?? nil
    }
    private func applySourceRetention(_ model: ModelManifest, loaded: SnapshotManifest) async {
        guard let provenance = loaded.derived else { return }
        let sourceBytes = provenance.sourceFiles?.reduce(0) { $0 + $1.bytes } ?? (model.weights.dtype == model.basePrecision ? model.weights.bytes?.values.reduce(0, +) ?? 0 : 0)
        guard
            SourceRetentionPolicy.eligible(
                sourcePrecision: model.basePrecision, derivedPrecision: loaded.precision, selectedPrecision: config.selections[model.id]?.precision ?? model.defaultPrecision,
                residentPrecision: loadedPreferences[model.id]?.precision,
                verifiedDerived: provenance.sourceDtype == model.basePrecision && provenance.sourceRevision == model.weights.revision, sourceBytes: sourceBytes,
                thresholdBytes: sourceRetentionThresholdBytes, getInFlight: gets[model.id] != nil), !Task.isCancelled
        else { return }
        do {
            if try await fetcher.store.retainVerifiedDerived(
                model: model.id, revision: model.weights.revision, sourcePrecision: model.basePrecision, derivedPrecision: loaded.precision,
                thresholdBytes: sourceRetentionThresholdBytes) != nil
            {
                await refreshInstalled(forceDownloadSizes: true)
            }
        } catch { emit(.error(id: model.id, message: "Loaded, but source cleanup was skipped: \(error.localizedDescription)")) }
    }
    func unloadResident(id: String, notifyChanges: Bool = true) async throws {
        beginUnload(id); defer { endUnload(id) }
        try requireIdleResident(id)
        if let revision = captureRevision(id) { dormantRevisions[id] = revision }
        timers[id]?.cancel(); timers[id] = nil; await supervisor.unload(id: id)
        if residents.removeValue(forKey: id) != nil { await fetcher.store.release(model: id) }
        clearResidentDescriptor(id)
        footprints[id] = nil; loadedPreferences[id] = nil; loadedRunners[id] = nil; modelRows[id]?.residentRunner = nil
        if let m = try? manifest(id) {
            for p in m.precisions where modelRows[id]?.states[p]?.state == .loaded {
                modelRows[id]?.states[p] = .init(state: .ready, phase: droppedCatalogModels[id] == nil ? nil : "catalogRemoved")
            }
        }
        for p in modelRows[id]?.precisions ?? [] where modelRows[id]?.states[p]?.phase == "localFolderErrorLoaded" {
            modelRows[id]?.states[p]?.phase = "localFolderError"
        }
        modelSettings.applied[id] = nil; modelRows[id]?.kernels = []; updatePending(id)
        signalMemoryChange()
        if notifyChanges { emit(.memory(memory())); changed() } else { activityAggregator.changed() }
    }
    public func setPrecision(id: String, precision: String) async throws {
        try admitWork(); await refreshLocalModels()
        if RetiredDtype.isRetired(precision) { throw Self.retiredRefusal(precision) }
        let model = try manifest(id)
        guard gets[id] == nil && loadTasks[id] == nil else {
            throw WorkerFailure(code: "busy", message: "Wait for Get or Load to finish before changing precision.")
        }
        guard model.precisions.contains(precision) else { throw WorkerFailure(code: "precision", message: "That dtype is not offered by \(id).") }
        guard let runner = selectedRunner(model), model.supports(dtype: precision, runner: runner.name) else {
            throw WorkerFailure(code: "unsupported_format", message: modelRows[id]?.dtypeUnavailableReason(precision) ?? "No runner in this Arsenal yet.")
        }
        if let reason = modelRows[id]?.dtypeUnavailableReason(precision) { throw WorkerFailure(code: "unsupported_format", message: reason) }
        var preference = config.selections[id] ?? .init(precision: model.defaultPrecision)
        preference.precision = precision; preference.explicitSelection = true
        config.selections[id] = preference
        try persist(); updateMetrics(model); updatePending(id); changed()
    }
    public func setKeepHot(id: String, on: Bool) throws {
        let m = try manifest(id)
        var preference = config.selections[id] ?? .init(precision: modelRows[id]?.selectedPrecision ?? m.defaultPrecision, explicitSelection: config.table.contains(id))
        preference.keepHot = on; config.selections[id] = preference
        try persist(); modelRows[id]?.keepHot = on; residents[id]?.keepHot = on; armTimer(id); signalMemoryChange(); changed()
    }
    public func deleteWithdrawnTiers(id: String) async throws {
        try admitWork()
        let m = try manifest(id)
        guard gets[id] == nil, loadTasks[id] == nil else { throw WorkerFailure(code: "busy", message: "Finish Get or Load before removing files.") }
        let withdrawn = await fetcher.store.withdrawnPrecisions(model: id, offered: Set(m.precisions)).keys.sorted()
        if let resident = loadedPreferences[id]?.precision, withdrawn.contains(resident) {
            throw WorkerFailure(code: "busy", message: "Unload \(modelRows[id]?.name ?? id) before removing its \(resident.uppercased()) files.")
        }
        for precision in withdrawn { try await fetcher.store.delete(model: id, precision: precision, residentPrecision: loadedPrecision(id), reason: "delete_withdrawn_precision") }
        do { try await reselectAfterDelete(id: id, deleted: withdrawn) } catch { emit(.error(id: id, message: error.localizedDescription)) }
        await refreshInstalled(forceDownloadSizes: true); changed()
    }
    public func delete(id: String) async throws {
        if droppedCatalogModels[id] == nil { _ = try manifest(id) }
        guard gets[id] == nil, loadTasks[id] == nil else { throw WorkerFailure(code: "busy", message: "Finish Get or Load before deleting.") }
        // Only weights and their caches go. The model folder (support/models/<id>) holds the signed definition and measurement
        // results, which are not weights and cannot be rebuilt, so it stays.
        endGateWait(id)
        try await unload(id: id); try await fetcher.store.delete(model: id, partialSHA256s: partialHashes(id: id))
        config.selections[id] = nil; try persist(); modelRows[id]?.states = [:]; await refreshInstalled(forceDownloadSizes: true); changed()
    }
    func performRun(id: String, op: String, params: JSONObject, progress: @escaping ProgressTransform, sink: RunSink?) async throws -> JSONObject {
        try admitRun(id)
        try await waitForRevisionSwap(id)
        pendingRuns[id, default: 0] += 1
        defer { pendingRuns[id, default: 0] -= 1; signalMemoryChange(); scheduleRevisionSwap(id) }
        var requestParams = try await validatedRequest(id: id, op: op, params: params)
        let operation = try await resolvedOperations(id: id).first { $0.name == op }
        let preparationDirectory = paths.directory("jobs").appendingPathComponent(UUID().uuidString)
        defer { if operation?.prepareImage != nil { try? FileManager.default.removeItem(at: preparationDirectory) } }
        if let preparation = operation?.prepareImage {
            requestParams = try await prepareOperationImage(id: id, preparation: preparation, request: requestParams, directory: preparationDirectory)
            // Validate the replacement through the same schema/file boundary before loading the target.
            requestParams = try await validatedRequest(id: id, op: op, params: requestParams)
        }
        if op == "delete_voice", usesVoicePersistence(id), let voice = requestParams["voice_id"]?.string {
            try await deleteSavedVoice(id: id, voiceID: voice); return ["deleted": .string(voice)]
        }
        // A Get or pending setting never replaces a healthy resident behind the user's back.
        if try await admitRequest(id: id, op: op, operationValidated: true) {
            try await ensureLoaded(id: id, precision: nil, fast: nil, residency: .onDemand)
            ActivityRequestContext.tracking?.finishLoading()
        }
        residents[id]?.busy += 1; timers[id]?.cancel(); timers[id] = nil
        defer { residents[id]?.busy -= 1; touch(id) }
        var request = requestParams
        let support = paths.support
        let dir = paths.directory("jobs").appendingPathComponent(UUID().uuidString)
        let output = operation?.output
        if needsOutputDirectory(id: id, op: op, output: output, request: request) {
            try await Task.detached { try TransientModelJobs.prepare(dir, support: support) }.value; request["out_dir"] = .string(dir.path)
        }
        do {
            ActivityRequestContext.tracking?.schedule()
            if let tracking = ActivityRequestContext.tracking { activityAggregator.phase(tracking.id, op == "chat" ? "prefill" : "step") }
            let ceiling = workerMemoryCeiling(
                id, requirement: residentMemoryRequirements[id], precision: loadedPreferences[id]?.precision, forced: forceLoadedResidents.contains(id))
            let result = try await supervisor.request(
                id: id, op: op, params: request, timeout: operation?.policy?.timeoutS ?? 180, firstResponseTimeout: operation?.policy?.firstResponseTimeoutS,
                progress: progress, sink: sink, memoryCeiling: ceiling)
            try RequestedOutputSchema.validate(result, operation: op, input: request)
            if op == "design_voice", usesVoicePersistence(id) {
                let input = request; try await Task.detached { try DesignedVoicePreviews.record(model: id, request: input, result: result, support: support) }.value
            }
            if residentManifests[id]?.schema == 3, let operation {
                return try OperationArtifacts.output(result, operation: operation, directory: request["out_dir"]?.string.map { URL(fileURLWithPath: $0) })
            }
            return result
        } catch {
            if await supervisor.hasExited(id: id) {
                forceLoadedResidents.remove(id)
                residents[id] = nil; footprints[id] = nil; residentLocalDigests[id] = nil; let loaded = loadedPreferences.removeValue(forKey: id); loadedRunners[id] = nil
                modelRows[id]?.residentRunner = nil; await fetcher.store.release(model: id)
                if let p = loaded?.precision ?? modelRows[id]?.selectedPrecision {
                    modelRows[id]?.states[p] = .init(state: .ready, error: error is CancellationError ? nil : error.localizedDescription)
                }; changed()
            }; throw error
        }
    }
    public func menuSections() async -> [LoadedMenuSection] { residents.keys.sorted().flatMap { residentMenus[$0] ?? [] } }
    public func menuActions() -> [LoadedAction] { residents.keys.sorted().flatMap { residentMenus[$0] ?? [] }.flatMap(\.items) }
    private func declaredMenuSections(_ model: ModelManifest) -> [LoadedMenuSection] {
        func actions(_ specs: [MenuSpec]) -> [LoadedAction] {
            specs.map { action in
                LoadedAction(
                    id: "\(model.id)-\(action.id)", modelID: model.id, title: action.title, op: action.op, input: action.input,
                    params: (action.params ?? [:]).mapValues { scalar in
                        switch scalar {
                        case .string(let v): return .string(v)
                        case .number(let v): return .number(v)
                        case .bool(let v): return .bool(v)
                        }
                    })
            }
        }
        var sections = (model.menu?.sections ?? []).enumerated().map { index, section in
            LoadedMenuSection(id: "\(model.id)-section-\(index)", modelID: model.id, title: section.title, items: actions(section.items))
        }
        if let legacy = model.menu?.actions, !legacy.isEmpty { sections.insert(.init(id: model.id, modelID: model.id, title: model.name, items: actions(legacy)), at: 0) }
        return sections
    }
    public func memory() -> MemorySnapshot {
        let total = footprints.values.reduce(0) { $0 + $1.footprintMB }
        return MemorySnapshot(availableMB: probe.noSwapAvailableMB(loadedMB: total), workersMB: total, models: footprints)
    }
    public func catalogStatus() -> CatalogStatus {
        let errors = [sourceError, localDiscoveryError].compactMap { $0 } + config.warnings
        return CatalogStatus(source: sourceName, pinnedCommit: catalog.sourceCommit, verified: true, error: errors.isEmpty ? nil : errors.joined(separator: " "))
    }
    func kernelStatus(_ id: String, _ details: JSONObject) {
        guard let kernels = details["kernels"]?.array?.compactMap(\.object), modelRows[id]?.kernels != kernels else { return }
        modelRows[id]?.kernels = kernels; changed()
    }
    func status(_ id: String, _ memory: WorkerMemory) { footprints[id] = memory; residents[id]?.memoryMB = memory.footprintMB; emit(.memory(self.memory())) }
    func persist() throws {
        let file = paths.support.appendingPathComponent("config.json")
        var candidate = config
        rememberCatalogIdentities(in: &candidate)
        var candidateSettings = modelSettings
        candidateSettings.discardInvalidSavedValues(); candidate.settings = candidateSettings.values
        do {
            let hadWarnings = !config.warnings.isEmpty
            let saved = try candidate.writeMerging(file, baseline: persistedConfig)
            config = saved; persistedConfig = saved; rejectedRevisionDigests = saved.rejectedRevisions
            configData = try? Data(contentsOf: file)
            for id in modelSettings.declarations.keys {
                if let warning = modelSettings.warning(id) { settingsWarnings[id] = warning }
            }
            for id in config.selections.keys { config.selections[id]?.fast = true }
            modelSettings.load(saved.settings); catalogEntryCache.removeAll()
            for model in allManifests {
                let id = model.id
                let preference = saved.selections[id] ?? .init(precision: model.defaultPrecision)
                if model.precisions.contains(preference.precision) { modelRows[id]?.selectedPrecision = servablePrecision(model, preference.precision) }
                let fast = modelRows[id]?.fastAvailable == true
                modelRows[id]?.fast = fast; modelRows[id]?.toolsShown = preference.toolsShown
                modelRows[id]?.keepHot = preference.keepHot; residents[id]?.keepHot = preference.keepHot
                updateRunnerRow(model)
                if residents[id] != nil { armTimer(id) }
                updatePending(id)
            }
            if hadWarnings || !config.warnings.isEmpty { config.clearWarningsAfterWrite(); emit(.catalog(catalogStatus())) }
            signalMemoryChange()
        } catch {
            let migratedTable = config.legacyTableNeedsMigration ? config.table : nil
            let migration = config.migrations["tableV2"]
            config = persistedConfig
            if let migratedTable { config.table = migratedTable; config.migrations["tableV2"] = migration }
            throw error
        }
    }
    func persistSelectionBestEffort(_ id: String) {
        do { try persist() } catch { emit(.error(id: id, message: "Selection could not be saved: \(error.localizedDescription)")) }
    }
    public func setResidencySettings(_ settings: ResidencySettings) async throws {
        guard KeepHot.choices.contains(settings.manualIdleMinutes), KeepHot.choices.contains(settings.onDemandIdleMinutes) else {
            throw WorkerFailure(code: "settings", message: "Unsupported Keep Hot window.")
        }; config.residency = settings; try persist(); for id in residents.keys { armTimer(id) }
    }
    public func residencySettings() async -> ResidencySettings { config.residency }
    func touch(_ id: String) { residents[id]?.lastUsed = Date().timeIntervalSince1970; armTimer(id); signalMemoryChange() }
    func armTimer(_ id: String) {
        timers[id]?.cancel(); timers[id] = nil
        guard let resident = residents[id], let deadline = ResidencyPolicy.deadline(resident, settings: config.residency, minuteSeconds: minuteSeconds) else { return }
        timers[id] = Task { [weak self] in do { try await Task.sleep(for: .seconds(max(0, deadline - Date().timeIntervalSince1970))); await self?.idleUnload(id) } catch {} }
    }
    private func idleUnload(_ id: String) async { if residents[id]?.busy == 0 { try? await unload(id: id) } }
    private func startPressureMonitor() { pressure = MemoryPressureMonitor { [weak self] in Task { await self?.shedPressure() } } }
    public func shedPressure() async {
        let victims = ResidencyPolicy.pressureVictims(Array(residents.values), order: allManifests.map(\.id))
        for id in victims {
            // Pressure must not revoke Load Anyway, cancel a pending run, or interrupt a replacement.
            guard !forceLoadedResidents.contains(id), loadTasks[id] == nil, swapTasks[id] == nil, !reloading.contains(id), unloading[id, default: 0] == 0,
                pendingRuns[id, default: 0] == 0
            else { continue }
            try? await unloadResident(id: id)
        }
    }
    public func shutdown() async {
        if let shutdownTask { await shutdownTask.value; return }
        shuttingDown = true; signalMemoryChange()
        localFolderWatcher?.stop(); localFolderWatcher = nil
        let swaps = Array(swapTasks.values)
        for swap in swaps { swap.cancel() }
        let gets = Array(self.gets.values), loads = Array(loadTasks.values), runs = Array(runTasks.values)
        for task in gets + loads { task.cancel() }; for task in runs { task.cancel() }
        for timer in timers.values { timer.cancel() }; timers.removeAll(); pressure = nil
        let supervisor = self.supervisor, artifacts = self.artifacts, support = paths.support
        let drain = Task {
            await withTaskGroup(of: Void.self) { group in
                group.addTask { await supervisor.shutdown() }
                group.addTask { await artifacts.shutdown() }
                for task in gets + loads { group.addTask { _ = try? await task.value } }
                for task in runs { group.addTask { _ = try? await task.value } }
                await group.waitForAll()
            }
            do { try await Task.detached { try TransientModelJobs.prune(support: support) }.value } catch {
                emit(.error(id: nil, message: "Expired preview cleanup skipped: \(error.localizedDescription)"))
            }
        }
        shutdownTask = drain; await drain.value
        for swap in swaps { await swap.value }
        await supervisor.shutdown()
        for id in Array(residents.keys) { await fetcher.store.release(model: id) }
        residentMenus.removeAll(); residentOperations.removeAll(); residentManifests.removeAll()
        self.gets.removeAll(); loadTasks.removeAll(); runTasks.removeAll(); getProgress.removeAll(); residents.removeAll(); loadedPreferences.removeAll()
        loadedRunners.removeAll(); residentLocalDigests.removeAll(); residentMemoryRequirements.removeAll()
        footprints.removeAll(); loadReservations.removeAll(); loadReservationReleases.removeAll(); forceLoadedResidents.removeAll()
        for id in modelRows.keys {
            modelRows[id]?.residentRunner = nil
            for precision in modelRows[id]?.precisions ?? [] where modelRows[id]?.states[precision]?.state == .loaded || modelRows[id]?.states[precision]?.state == .loading {
                modelRows[id]?.states[precision] = .init(state: .ready)
            }
        }
        emit(.memory(memory())); changed()
    }
}

extension RealModelsService {
    public func getPlan(id: String, dtype: String) async throws -> GetPlan {
        try requireCurrentCatalogModel(id)
        try admitWork(); await refreshLocalModels()
        if RetiredDtype.isRetired(dtype) { throw Self.retiredRefusal(dtype) }
        let model = try manifest(id)
        guard model.precisions.contains(dtype) else { throw WorkerFailure(code: "precision", message: "That dtype is not offered by \(id).") }
        guard let runner = selectedRunner(model), model.supports(dtype: dtype, runner: runner.name) else {
            throw WorkerFailure(code: "unsupported_format", message: modelRows[id]?.dtypeUnavailableReason(dtype) ?? "No runner in this Arsenal yet.")
        }
        let pending = try await availability(model, precision: dtype)
        var name = model.name + " weights and assets"
        // A removed 16-bit copy is restored before any tier is made from it; the plan names that, not just a size.
        if pending.networkBytes + pending.localCopyBytes > 0, model.vendor?[dtype] == nil, !snapshotPresent(model, model.basePrecision),
            let receipt = try? await fetcher.store.sourceRetentionReceipt(model: model.id, revision: model.weights.revision, sourcePrecision: model.basePrecision)
        {
            name += ", restoring the 16-bit copy removed while \(receipt.derivedPrecision) was in use"
        }
        var items: [GetPlanItem] = [
            .init(
                id: "weights", name: name, kind: "weights", downloadBytes: pending.networkBytes, localCopyBytes: pending.localCopyBytes,
                networkContent: pending.networkContent, localCloneBytes: pending.localCloneBytes)
        ]
        try artifacts.validateEngine(model: model, runner: runner)
        items.append(.init(id: "engine", name: (runner.worker ?? runner.name) + " engine (included in Arsenal)", kind: "engine", downloadBytes: 0))
        return GetPlan(id: id, dtype: dtype, runner: runner.name, items: items, identity: try getPlanIdentity(model))
    }
    /// On-demand metadata refresh. Existing workers, requests, code snapshots and loaded menu contracts stay intact.
    /// The busy refusal's text (the table recognises it to drop a stale one once Refresh is allowed again).
    public static let refreshBusyMessage = "Finish Get or Load before refreshing models."
    var refreshBusy: String? { gets.isEmpty && loadTasks.isEmpty ? nil : Self.refreshBusyMessage }
    public func refreshRefusal() async -> String? { refreshBusy }
    public func refresh() async throws {
        try admitWork()
        if let refusal = refreshBusy { throw WorkerFailure(code: "busy", message: refusal) }
        refreshing = true
        defer { refreshing = false; for id in residents.keys { scheduleRevisionSwap(id) } }
        var nextConfig = try ArsenalConfiguration.read(paths.support.appendingPathComponent("config.json"))
        var nextSource = source, nextSourceName = sourceName
        let settings = nextConfig.catalog["source"]?.object ?? [:]
        if let path = settings["path"]?.string {
            let local = LocalGitCatalogSource(repo: URL(fileURLWithPath: path)); try local.validateAvailability()
            nextSource = local; nextSourceName = path
        } else if let base = settings["rawBase"]?.string {
            guard let url = URL(string: base), url.scheme == "https" || (url.scheme == "http" && ["localhost", "127.0.0.1", "::1"].contains(url.host ?? "")) else {
                throw ManifestError.invalid("catalog source must be HTTPS or loopback HTTP")
            }
            nextSource = HTTPSCatalogSource(rawBase: url); nextSourceName = base
        } else if nextConfig.catalog["source"]?.string == "bundle" {
            guard let resources = (source as? BundleCatalogSource)?.resources ?? bundledResources else { throw ManifestError.invalid("bundled catalog source is unavailable") }
            nextSource = try CatalogBootstrap.verifiedBundleSource(resources: resources); nextSourceName = "bundle"
        } else if let label = nextConfig.catalog["source"]?.string {
            throw ManifestError.invalid("unknown catalog source \(label)")
        }
        let directory =
            (nextSource as? BundleCatalogSource)?.resources.appendingPathComponent("catalog") ?? (nextSource as? LocalGitCatalogSource)?.repo.appendingPathComponent(
                "Resources/catalog")
            ?? catalogDirectory ?? paths.directory("catalog")
        let selected = try CatalogBootstrap.select(bundledDirectory: directory, cachedDirectory: paths.directory("catalog"), publicKey: publicKey)
        guard selected.document.sequence >= catalog.sequence else { throw ManifestError.invalid("catalog rollback rejected") }
        // A failed preparation must not advance the live sequence or invalidate its current trusted kinds.
        let staging = paths.cache.appendingPathComponent("refresh-" + UUID().uuidString)
        defer { if FileManager.default.fileExists(atPath: staging.path) { try? OwnedCacheCleanup.remove([staging], support: paths.support, operation: "refresh_staging") } }
        let verifier = ModelFolderInstaller(support: staging, kindsDirectory: (nextSource as? BundleCatalogSource)?.kindsDirectory ?? kindsDirectory)
        _ = try await verifier.accept(data: selected.data, signature: selected.signature, publicKey: publicKey)
        let preparedKinds = try await verifier.prepareKinds(source: nextSource)
        let nextKinds = preparedKinds == nil ? specs : try KindSpec.load(directory: preparedKinds!)
        try await installer.acceptPrepared(data: selected.data, signature: selected.signature, publicKey: publicKey, kindsDirectory: preparedKinds)
        let nextKindsDirectory = try await installer.verifiedKindsDirectory()
        nextConfig = try ArsenalConfiguration.read(paths.support.appendingPathComponent("config.json"))
        rememberDroppedCatalogModels(next: selected.document)
        catalog = selected.document; source = nextSource; sourceName = nextSourceName; sourceError = nil; config = nextConfig; persistedConfig = nextConfig
        rejectedRevisionDigests = config.rejectedRevisions
        configData = try? Data(contentsOf: paths.support.appendingPathComponent("config.json"))
        kindsDirectory = nextKindsDirectory; specs = nextKinds
        await fetcher.setLocalSources(nextConfig.get["localSources"]?.array?.compactMap(\.string))
        operationCache.removeAll(); operationFailures.removeAll(); catalogEntryCache.removeAll(); installedRowCache.removeAll(); installationUpdates.removeAll()
        intactMemo.removeAll()
        lastLocalDiscovery = nil
        pruneUnlistedRows()
        try await initializeRows(preservingActive: true)
        config = try ArsenalConfiguration.read(paths.support.appendingPathComponent("config.json"))
        rejectedRevisionDigests = config.rejectedRevisions
        configData = try? Data(contentsOf: paths.support.appendingPathComponent("config.json"))
        persistedConfig = config
        modelSettings.load(config.settings)
        for model in allManifests {
            let id = model.id, preferences = config.selections[id] ?? .init(precision: model.defaultPrecision)
            modelRows[id]?.selectedPrecision = servablePrecision(model, preferences.precision)
            modelRows[id]?.keepHot = preferences.keepHot; residents[id]?.keepHot = preferences.keepHot
            updateRunnerRow(model); updatePending(id)
            if residents[id] != nil { armTimer(id) }
        }
        emit(.catalog(catalogStatus())); changed()
        // Orphans and staging only: a model the new catalog drops keeps its weights until the user removes them.
        await collectWeights(startup: false)
    }
}

extension RealModelsService {
    private func getPlanIdentity(_ model: ModelManifest) throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var data = try encoder.encode(model)
        if let folder = localModels[model.id] {
            data.append(Data(folder.contentDigest.utf8))
        } else if let item = catalog.models.first(where: { $0.id == model.id }) {
            data.append(try encoder.encode(item.files))
        }
        return FileDigest.sha256(data)
    }
}

extension RealModelsService {
    public func setSettings(id: String, values: JSONObject) async throws {
        try admitWork(); let model = try manifest(id), write = try ModelSettings.splitMenuPlacement(values)
        var candidate = modelSettings; for key in write.settings.keys.sorted() { try candidate.set(id, key: key, value: write.settings[key]!) }
        let oldSettings = modelSettings, oldConfig = config; modelSettings = candidate; config.settings = candidate.values
        if let shown = write.shown {
            var preference = config.selections[id] ?? .init(precision: modelRows[id]?.selectedPrecision ?? model.defaultPrecision, explicitSelection: config.table.contains(id))
            preference.toolsShown = shown; config.selections[id] = preference
        }
        do { try persist() } catch { modelSettings = oldSettings; config = oldConfig; throw error }
        if let shown = write.shown { modelRows[id]?.toolsShown = shown; catalogEntryCache[id] = nil }
        settingsWarnings[id] = nil; updatePending(id); changed()
    }
    public func setToolsShown(id: String, on: Bool) async throws {
        let model = try manifest(id)
        var preference = config.selections[id] ?? .init(precision: modelRows[id]?.selectedPrecision ?? model.defaultPrecision, explicitSelection: config.table.contains(id))
        preference.toolsShown = on; config.selections[id] = preference; try persist()
        modelRows[id]?.toolsShown = on; catalogEntryCache[id] = nil; changed()
    }
    public func reference(id: String, dtype: String? = nil, mode: String? = nil) async throws -> ModelReference? {
        let model = try manifest(id), row = modelRows[id]
        let dtype = dtype ?? row?.selectedPrecision ?? model.defaultPrecision, mode = mode ?? (row?.fast == true ? "fast" : "exact")
        guard model.precisions.contains(dtype), ["fast", "exact"].contains(mode) else {
            throw WorkerFailure(code: "invalid_request", message: "Choose an offered dtype and fast or exact mode.")
        }
        return ReferenceFigures.best(resultCache[id] ?? [], chip: Self.chip, dtype: dtype, mode: mode)
    }
}
