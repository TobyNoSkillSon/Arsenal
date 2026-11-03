// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import ArsenalBench
import ArsenalWire

actor APIRouter {
    var controlGets: [String: ControlGetProgress] = [:]
    let dialects = DialectRegistry.standard
    let service: any ModelsService
    let support: URL
    let benchmarkController: BenchmarkController?
    var port = 0
    var mcpSessions: [String: MCPSession] = [:]
    var mcpJobs: [String: MCPJob] = [:]
    var mcpOutputs: [String: URL] = [:]
    var mcpOutputOrder: [String] = []
    /// Clock, disk and memory readings for the MCP surface; tests swap in a simulated environment.
    var mcpEnvironment = MCPEnvironment.live
    /// Jobs persisted across restarts (`mcp-jobs.json`), terminal ones kept 24 h.
    var mcpJobRecords: [String: MCPJobRecord] = [:]
    /// One Get per model and dtype; a second MCP call joins it.
    var mcpGets: [String: MCPGet] = [:]
    /// The size last offered for consent per model/dtype, for the 10% re-ask rule.
    var mcpReportedSizes: [String: (bytes: Int64, at: Date)] = [:]
    var mcpRates: MCPRateHistory { MCPRateHistory(file: support.appendingPathComponent("mcp-download-rates.json")) }
    func configureMCP(_ environment: MCPEnvironment) { mcpEnvironment = environment }
    let playbackOverride: PlaybackQueue?
    var mcpToken: String?
    var playback: PlaybackQueue { get async { if let playbackOverride { return playbackOverride }; return await PlaybackQueue.shared } }
    var previews: [String: VoicePreview] = [:]
    var previewOrder: [String] = []
    var predictions: [String: PredictionRecord] = [:]
    static let terminalPredictionCacheLimit = 128
    static let activePredictionLimit = 32
    var terminalPredictionOrder: [String] = []
    var predictionTasks: [String: Task<Void, Never>] = [:]
    var recoveredPredictions = false
    var operationJobs: [String: OperationJob] = [:]
    var operationJobTasks: [String: Task<Void, Never>] = [:]
    init(service: any ModelsService, support: URL, benchmarkController: BenchmarkController? = nil, playbackQueue: PlaybackQueue? = nil) {
        self.service = service; self.support = support; self.benchmarkController = benchmarkController; self.playbackOverride = playbackQueue
    }
    func setPort(_ port: Int) { self.port = port }
    static func releaseVersion(bundle: Bundle = .main) -> String {
        bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "development"
    }
    var origin: String { "http://127.0.0.1:\(port)" }
    var jobsRoot: URL { support.appendingPathComponent("jobs", isDirectory: true) }
    func handle(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        try await ActivityRequestContext.$requester.withValue(ActivityRequestContext.identity(headers: request.headers)) {
            let object = try? WireJSON.decode(request.body)
            if let force = object?["force"], force.bool == nil { throw HTTPFailure(422, "force must be a boolean.") }
            let forceHeader = request.headers["x-arsenal-force-load"] ?? request.headers["x-mobius-force-load"] // legacy-compat: clients written before the rename
            let force = object?["force"]?.bool == true || forceHeader == "true"
            let body = object?["force"] == nil ? request.body : try RawJSON(request.body).replacing(removing: ["force"])
            let forwarded = HTTPRequest(method: request.method, target: request.target, headers: request.headers, body: body)
            try await LoadMemoryContext.$force.withValue(force) { try await handleAttributed(forwarded, connection) }
        }
    }
    private func handleAttributed(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        do {
            if request.path == "/mcp" { try await mcp(request, connection); return }
            if request.path == "/arsenal/v1/mcp/connect" { try await mcpConnect(request, connection); return }
            if request.path == "/arsenal/v1/playback" { try await playbackControl(request, connection); return }
            if request.path.hasPrefix("/arsenal/files/") {
                guard request.method == "GET" else { throw HTTPFailure(405, "Use GET.") }
                try await file(request, connection); return
            }
            if request.path.hasPrefix("/arsenal/v1/jobs/") { try await operationJobRoute(request, connection); return }
            if let capability = CapabilityContract.all.first(where: { request.path == "/arsenal/v1/" + $0.route }) {
                try await capabilityAlias(request, connection, operation: capability.operation); return
            }
            if request.path.hasPrefix("/arsenal/v1/") { try await control(request, connection); return }
            guard let dialect = dialects.module(for: request.path) else { throw HTTPFailure(404, "Endpoint is not supported by Arsenal.") }
            try await dialect.handle(request, connection: connection, router: self)
        } catch let error as WorkerFailure {
            throw serviceHTTPFailure(error)
        }
    }
    func openai(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        switch request.path {
        case "/v1/models":
            guard request.method == "GET" else { throw HTTPFailure(405, "Use GET.") }
            try await connection.respond([
                "object": "list", "data": await rows().filter { request.query["kind"] == nil || $0["kind"] as? String == request.query["kind"] }.map(openAIModel)
            ])
        case let path where path.hasPrefix("/v1/models/"):
            guard request.method == "GET" else { throw HTTPFailure(405, "Use GET.") }
            let id = String(path.dropFirst("/v1/models/".count))
            guard let row = await rows().first(where: { $0["id"] as? String == id }) else { throw HTTPFailure(404, "Model is not in the Arsenal table.") }
            try await connection.respond(openAIModel(row))
        case "/v1/audio/speech": try await speech(request, connection)
        case "/v1/images/generations": try await images(request, connection)
        case "/v1/chat/completions": try await chat(request, connection)
        default: throw HTTPFailure(404, "Endpoint is not supported by Arsenal.")
        }
    }
    func object(_ request: HTTPRequest) throws -> JSONObject {
        guard request.method == "POST" else { throw HTTPFailure(405, "Use POST.") }
        guard request.headers["content-type"]?.lowercased().hasPrefix("application/json") == true else { throw HTTPFailure(415, "Send application/json.") }
        guard let value = try? WireJSON.decode(request.body) else { throw HTTPFailure(400, "Body must be a JSON object.") }
        return value
    }
    func required(_ body: JSONObject, _ key: String) throws -> String {
        guard let value = body[key]?.string, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw HTTPFailure(422, "\(key) must be a nonempty string.") }
        return value
    }
    func reject(_ body: JSONObject, _ keys: [String]) throws {
        for key in keys where body[key] != nil { throw HTTPFailure(400, "\(key) is not supported by this model profile.") }
    }
    func rows() async -> [[String: Any]] {
        let encoded = try? JSONEncoder().encode(await service.rows())
        return encoded.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [[String: Any]] } ?? []
    }
    /// A model's row whether or not it is on the user's table: agents and scripts use downloaded models off it.
    func anyRow(_ id: String) async -> ModelRow? {
        if let row = await service.rows().first(where: { $0.id == id }) { return row }
        return await service.catalog(query: .init()).first { $0.id == id }?.row
    }
    func catalogRows() async -> [[String: Any]] {
        let encoded = try? JSONEncoder().encode(await service.catalog(query: CatalogQuery()).map(\.row))
        return encoded.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [[String: Any]] } ?? []
    }
    /// Installed rows, including downloaded models an agent uses off the table (catalog rows the table and residents do not list).
    func installedModels() async -> [[String: Any]] {
        let listed = await rows()
        let ids = Set(listed.compactMap { $0["id"] as? String })
        let offTable = await service.catalog(query: CatalogQuery()).filter { $0.installed && !ids.contains($0.id) }.compactMap { encodedRow($0.row) }
        return installed(listed + offTable)
    }
    private func encodedRow(_ row: ModelRow) -> [String: Any]? {
        (try? JSONEncoder().encode(row)).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
    }
    /// Rows with an installed precision. Takes rows already fetched, so one request needs one `rows()` snapshot.
    func installed(_ rows: [[String: Any]]) -> [[String: Any]] {
        rows.filter { row in
            let states = row["states"] as? [String: Any] ?? row["stateByPrecision"] as? [String: Any] ?? [:]
            return states.values.contains { state in
                let value = (state as? [String: Any])?["state"] as? String ?? state as? String ?? ""
                return ["ready", "loaded", "loading"].contains(value) || (state as? [String: Any])?["phase"] as? String == "localFolderErrorLoaded"
            } || (row["state"] as? String).map { ["ready", "loaded", "loading"].contains($0) } == true
        }
    }
    func resolve(_ id: String, kind: String?, operation: String? = nil) async throws -> [String: Any] {
        // One snapshot: every rows() refreshes local folders and installed state (about 1-2 ms on the request path).
        var all = await rows()
        // A model off the table answers from its catalog row (anyRow's fallback), so its operations and install check see it.
        if !all.contains(where: { $0["id"] as? String == id }),
            let found = await service.catalog(query: .init()).first(where: { $0.id == id }), let encoded = encodedRow(found.row)
        {
            all.append(encoded)
        }
        guard let row = all.first(where: { $0["id"] as? String == id }), kind == nil || row["kind"] as? String == kind else {
            throw HTTPFailure(404, "Unknown \(kind ?? "local") model \(id); call \(MCPNames.prefix)find.")
        }
        if let operation {
            // Tool routes and the menu ask the service's request admission, including resident/recovery bypasses.
            if let refusal = await service.runRefusal(id: id, op: operation) { throw refusal }
        } else if !installed(all).contains(where: { $0["id"] as? String == id }) {
            throw HTTPFailure(404, "Model \(id) is not installed; \(MCPNames.prefix)download(model: \(id)) downloads \(downloadSize(row)).")
        }
        return row
    }
    func downloadSize(_ row: [String: Any]) -> String {
        let selected = row["selectedPrecision"] as? String ?? "bf16"
        let bytes = (row["downloadBytes"] as? [String: Any])?[selected] as? Double ?? row["sizeBytes"] as? Double
        guard let bytes else { return "weights (size unknown; call \(MCPNames.prefix)find)" }
        return bytes >= 1_000_000_000 ? String(format: "%.1f GB", bytes / 1e9) : String(format: "%.1f MB", bytes / 1e6)
    }
    func defaultModel(kind: String) async throws -> String {
        guard let id = await installedModels().first(where: { $0["kind"] as? String == kind })?["id"] as? String else {
            throw HTTPFailure(404, "No installed \(kind) model; call \(MCPNames.prefix)find then \(MCPNames.prefix)download(model: ID).")
        }; return id
    }
    func openAIModel(_ row: [String: Any]) -> [String: Any] {
        ["id": row["id"] ?? "", "object": "model", "created": 0, "owned_by": "local", "kind": row["kind"] ?? ""]
    }
    func run(_ id: String, op: String, params: JSONObject, sink: RunSink? = nil) async throws -> JSONObject {
        try Task.checkCancellation()
        let result = try await service.run(id: id, op: op, params: params, sink: sink)
        try RequestedOutputSchema.validate(result, operation: op, input: params)
        return result
    }
    func newJob() throws -> URL {
        let dir = jobsRoot.appendingPathComponent(UUID().uuidString.lowercased(), isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700]); return dir
    }
    func outputPath(_ path: String) throws -> URL {
        let url = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
        let root = jobsRoot.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        guard url.path.hasPrefix(root), !url.hasDirectoryPath else { throw HTTPFailure(500, "Runner output is outside its job directory.") }; return url
    }
    func outputURL(_ path: String) throws -> String {
        let url = try outputPath(path)
        let root = jobsRoot.resolvingSymlinksInPath().standardizedFileURL.path
        let relative = String(url.path.dropFirst(root.count + 1))
        let allowed = CharacterSet.urlPathAllowed.subtracting(CharacterSet(charactersIn: "%?#"))
        return origin + "/arsenal/files/" + (relative.addingPercentEncoding(withAllowedCharacters: allowed) ?? "")
    }
    func file(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let relative = String(request.path.dropFirst("/arsenal/files/".count)).removingPercentEncoding ?? ""
        guard !relative.isEmpty, !relative.split(separator: "/").contains(".."), !relative.contains("\\"), !relative.contains("\0") else {
            throw HTTPFailure(404, "File not found.")
        }
        let url = try outputPath(jobsRoot.appendingPathComponent(relative).path)
        guard !url.path.contains("/replicate/"), (try? OperationArtifacts.confined(url.path, roots: [jobsRoot], code: "invalid_output")) != nil,
            let size = try? FileDigest.bytes(url), size <= Int64(OperationArtifacts.maxArtifactBytes)
        else { throw HTTPFailure(404, "Output file not found.") }
        // Job files are streamed from disk: they are artifacts, not inline payloads, so the 64 MiB inline cap does not apply.
        try await connection.respondFile(url, size: size, type: contentType(url))
    }
}

func plain(_ value: JSONValue) -> Any {
    switch value {
    case .null: return NSNull()
    case .bool(let v): return v
    case .number(let v): return v
    case .string(let v): return v
    case .array(let v): return v.map(plain)
    case .object(let v): return v.mapValues(plain)
    }
}
func plain(_ object: JSONObject) -> [String: Any] { object.mapValues(plain) }
