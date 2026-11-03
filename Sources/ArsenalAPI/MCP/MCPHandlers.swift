// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import MCP

/// Delivers one progress notification: strictly increasing `progress`, optional `total`, a human message.
typealias MCPNotify = @Sendable (_ progress: Double, _ total: Double?, _ message: String) async -> Void

extension APIRouter {
    /// Every tool answers with text written for the agent; failures are `[category]` lines with `isError`, never thrown,
    /// except cancellation, which the transport turns into a cancelled request.
    func callMCPTool(_ name: String, arguments body: JSONObject, requester: String, server: MCP.Server?, progress: ProgressToken?) async throws -> CallTool.Result {
        var notify: MCPNotify?
        if let server, let progress {
            notify = { [weak server] value, total, message in
                try? await server?.notify(ProgressNotification.message(.init(progressToken: progress, progress: value, total: total, message: message)))
            }
        }
        return try await callMCPTool(name, arguments: body, requester: requester, notify: notify)
    }
    /// The same call with progress delivered to `notify` (tests collect it directly).
    func callMCPTool(_ name: String, arguments body: JSONObject, requester: String, notify: MCPNotify?) async throws -> CallTool.Result {
        do {
            let bare = name.hasPrefix(MCPNames.prefix) ? String(name.dropFirst(MCPNames.prefix.count)) : ""
            switch bare {
            case "find": return try await mcpFind(body)
            case "model": return try await mcpModel(body)
            case "status": return try await mcpStatus(body)
            case "add": return try await mcpAdd(body)
            case "remove": return try await mcpRemove(body)
            case "delete": return try await mcpDelete(body)
            case "download": return try await mcpDownload(body, requester: requester, notify: notify)
            case "load": return try await mcpLoad(body, requester: requester, notify: notify)
            case "unload": return try await mcpUnload(body)
            case "run": return try await mcpRunTool(body, requester: requester, notify: notify)
            case "wait": return try await mcpWaitTool(body, notify: notify)
            case "cancel": return try await mcpCancel(body)
            case "upload": return try mcpUpload(body)
            case "settings": return try await mcpSettings(body)
            case "playback": return try await mcpPlayback(body)
            default: throw MCPFailure(.invalidInput, "No tool \(name); tools/list has the 15 Arsenal tools.")
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let verbs = [
                "add": "Not added", "remove": "Not removed", "delete": "Not deleted", "download": "Not downloaded", "load": "Not loaded",
                "unload": "Not unloaded", "settings": "Not changed", "playback": "Playback unchanged", "cancel": "Not cancelled"
            ]
            return await mcpFailure(error, entry: nil, verb: verbs[String(name.dropFirst(MCPNames.prefix.count))] ?? "Not run").result
        }
    }
    /// A local model whose code nobody approved neither downloads, loads nor runs over MCP.
    func mcpRequireApproved(_ entry: CatalogEntry, verb: String) throws {
        guard entry.row.codeConsentRequired else { return }
        throw MCPFailure(
            .approval,
            "\(verb): \(entry.name) runs local code nobody has approved yet. Only with the user's yes: \(MCPNames.prefix)settings(model:\"\(entry.id)\", action:\"approve_code\").")
    }
    func mcpModelID(_ body: JSONObject) throws -> String {
        guard let id = body["model"]?.string?.trimmingCharacters(in: .whitespaces), !id.isEmpty else {
            throw MCPFailure(.invalidInput, "model must be a model id; \(MCPNames.prefix)find lists them.")
        }
        return id
    }
    /// The dtype a call asks for, checked against what the model offers.
    func mcpDtype(_ body: JSONObject, entry: CatalogEntry) throws -> String {
        let dtype = body["dtype"]?.string ?? entry.row.selectedPrecision
        guard entry.dtypes.contains(dtype) else {
            throw MCPFailure(.unsupported, "\(entry.name) offers \(entry.dtypes.joined(separator: ", ")), not \(dtype).")
        }
        if let reason = entry.row.dtypeUnavailableReason(dtype) { throw MCPFailure(.unsupported, "\(entry.name) at \(dtype) is unavailable: \(reason)") }
        return dtype
    }
    func mcpRunningJob(model: String) async -> MCPJob? {
        for job in mcpJobs.values.sorted(by: { $0.created < $1.created }) where job.model == model { if await job.status == .running { return job } }
        return nil
    }
    func mcpGetJob(model: String, dtype: String) async -> MCPJob? {
        for job in mcpJobs.values.sorted(by: { $0.created < $1.created }) where job.model == model && job.dtype == dtype {
            if await job.status == .running, await job.hasDownloaded, [MCPJob.Phase.downloading, .deriving, .queued].contains(await job.phase) { return job }
        }
        return nil
    }

    // MARK: Download, load, run

    /// Decides whether a Get may start: nil to proceed (or join a running Get), else the consent answer to return.
    private func mcpGetGate(_ availability: MCPAvailability, body: JSONObject, verb: MCPConsentVerb) async throws -> CallTool.Result? {
        let key = availability.entry.id + "/" + availability.dtype
        // A gated model waiting for the user: answer from the wait, so polling neither spends its tries nor reopens pages.
        // Once Arsenal gave up, a consented download is a fresh try, as a click on Get is (the token or access may have changed).
        if let wait = await anyRow(availability.entry.id)?.states[availability.dtype]?.gate, !(wait.gaveUp && body["allow_download"]?.bool == true) {
            throw MCPFailure(.waitingOnPerson, GatedModelText.line(wait, name: availability.entry.name))
        }
        if availability.inProgress || mcpGets[key] != nil { return nil }
        try mcpRequireDisk(availability)
        let allowed = body["allow_download"]?.bool == true
        if !allowed {
            if mcpIsFreeImport(availability) { return nil }
            mcpRemember(availability)
            return mcpText(await mcpConsentText(availability, verb: verb))
        }
        if let changed = mcpPlanChanged(availability) {
            mcpRemember(availability)
            let note = "The download changed since it was offered: now \(MCPFormat.size(changed.new)), was \(MCPFormat.size(changed.old))."
            return mcpText(await mcpConsentText(availability, verb: verb, prefix: note))
        }
        return nil
    }
    /// The download part of a job: joins a running Get of the same model and dtype, follows one started in the app, or
    /// starts one with the plan the user agreed to.
    func mcpAcquire(_ job: MCPJob, _ availability: MCPAvailability) async throws {
        let id = availability.entry.id, dtype = availability.dtype, key = id + "/" + dtype, environment = mcpEnvironment, service = service
        await job.expectDownload(
            total: availability.plan.map { ($0.totalDownloadBytes ?? 0) + $0.items.reduce(0) { $0 + $1.localCopyBytes } }, networkBytes: availability.networkBytes)
        await job.enter(.downloading)
        let observer = Task {
            let events = await service.events()
            var last = Date.distantPast
            func sample() async { if let state = await self.anyRow(id)?.states[dtype] { await job.download(state) } }
            await sample()
            for await event in events {
                guard !Task.isCancelled else { break }
                switch event {
                case .progress(let model, _, _) where model == id: break
                case .rowsChanged: break
                default: continue
                }
                let now = environment.now()
                guard now.timeIntervalSince(last) >= 1 else { continue }
                last = now; await sample()
            }
        }
        defer { observer.cancel() }
        if mcpGets[key] == nil, !availability.inProgress {
            // A shared Get that already finished for another call leaves nothing to do.
            let installed: Set<ModelState> = [.ready, .loading, .loaded]
            let now = await anyRow(id)?.states[dtype]?.state ?? .notInstalled
            if !installed.contains(availability.state), installed.contains(now) { await job.finishDownload(); return }
        }
        do {
            if let shared = mcpJoinGet(availability, waiter: job.id) {
                defer { if mcpGets[key]?.task == shared { mcpGets[key] = nil } }
                try await shared.value
            } else {
                try await mcpAwaitExternalGet(availability)
            }
        } catch {
            let status = await job.status
            if error is CancellationError || error is MCPFailure || Task.isCancelled || status != .running { throw error }
            if let gate = error as? HFGateError { throw MCPFailure(.waitingOnPerson, GatedModelText.line(gate, name: availability.entry.name)) }
            // A pinned file the repository lacks: retrying cannot help, so no "call again to resume".
            if let missing = error as? HFMissingFileError {
                throw MCPFailure(
                    .download, "Download of \(availability.entry.name) failed: \(missing.localizedDescription) Do not call download again; the model's pins must change.")
            }
            let final = ["code_consent", "local_folder", "pinned_file_missing", "busy", "unsupported_format", "precision", "not_found", "updating", "shutting_down"]
            if let failure = error as? WorkerFailure, final.contains(failure.code) { throw error }
            let total = await job.totalBytes, done = await job.bytes
            let at = total.map { " at " + MCPFormat.progress(done, of: $0) } ?? ""
            let reason = (error as? WorkerFailure)?.message ?? (error as? HTTPFailure)?.message ?? error.localizedDescription
            throw MCPFailure(
                .download,
                "Download of \(availability.entry.name) failed\(at): \(reason) Call \(MCPNames.prefix)download(model:\"\(id)\", allow_download:true) to resume; the user already agreed."
            )
        }
        if let state = await anyRow(id)?.states[dtype] { await job.download(state) }
        await job.finishDownload()
        if let measured = await job.downloadMeasurement() { mcpRates.record(host: mcpHost(availability.entry), bytes: measured.bytes, seconds: measured.seconds) }
    }
    /// Follows a Get started in the app or over HTTP until it ends.
    func mcpAwaitExternalGet(_ availability: MCPAvailability) async throws {
        let id = availability.entry.id, dtype = availability.dtype
        let events = await service.events()
        func state() async -> ModelState { await anyRow(id)?.states[dtype]?.state ?? .notInstalled }
        var current = await state()
        if current == .downloading || current == .deriving {
            for await _ in events {
                try Task.checkCancellation()
                current = await state()
                if current != .downloading && current != .deriving { break }
            }
        }
        try Task.checkCancellation()
        if current == .failed, let wait = await anyRow(id)?.states[dtype]?.gate {
            throw MCPFailure(.waitingOnPerson, GatedModelText.line(wait, name: availability.entry.name))
        }
        if current == .failed, let state = await anyRow(id)?.states[dtype], let final = Self.mcpFinalGetFailure(state, name: availability.entry.name) { throw final }
        guard [.ready, .loading, .loaded].contains(current) else {
            throw MCPFailure(
                .download,
                "The download of \(availability.entry.name) started outside this call stopped (\(current.rawValue)). With the user's yes, retry with allow_download:true.")
        }
    }
    /// A Get that ended on a pinned file its repository lacks (or a folder that became an error row): no retry can download
    /// it, so the agent is told not to call again rather than given the resume call.
    static func mcpFinalGetFailure(_ state: PrecisionState, name: String) -> MCPFailure? {
        guard state.state == .failed, let phase = state.phase, phase == RealModelsService.pinnedFileMissing || phase.hasPrefix("localFolderError") else { return nil }
        return MCPFailure(
            .download, "Download of \(name) failed: \(state.error ?? "a pinned file is not in its repository.") Do not call download again; the model's pins must change.")
    }
    /// Starts the MCP Get for this model and dtype, or joins the running one, and counts `waiter` on it. No suspension
    /// between the check and the registration, so concurrent calls share one Get. Nil for a Get started in the app or
    /// over HTTP, which MCP only follows.
    func mcpJoinGet(_ availability: MCPAvailability, waiter: String) -> Task<Void, any Error>? {
        let id = availability.entry.id, dtype = availability.dtype, key = id + "/" + dtype, service = service, plan = availability.plan
        if var get = mcpGets[key] {
            get.waiters.insert(waiter); mcpGets[key] = get; return get.task
        }
        guard !availability.inProgress else { return nil }
        let task = Task { if let plan { try await service.get(id: id, precision: dtype, approvedPlan: plan) } else { try await service.get(id: id, precision: dtype) } }
        mcpGets[key] = MCPGet(task: task, waiters: [waiter])
        return task
    }
    /// A cancelled job leaves its Get. The Get stops only when MCP started it, no other job waits on it and the job was
    /// still downloading; a Get started in the app or over HTTP is stopped only by an explicit cancel of the model.
    func mcpJobCancelled(_ job: MCPJob, model: String, dtype: String) async {
        let key = model + "/" + dtype, phase = await job.phase
        guard var get = mcpGets[key], get.waiters.contains(job.id) || phase == .queued else { return }
        get.waiters.remove(job.id); mcpGets[key] = get
        guard get.waiters.isEmpty, [MCPJob.Phase.queued, .downloading, .deriving].contains(phase) else { return }
        mcpGets[key] = nil; get.task.cancel()
        await service.cancelGet(id: model)
    }

    func mcpDownload(_ body: JSONObject, requester: String, notify: MCPNotify?) async throws -> CallTool.Result {
        let entry = try await mcpEntry(mcpModelID(body)), dtype = try mcpDtype(body, entry: entry)
        try mcpRequireApproved(entry, verb: "Not downloaded")
        try mcpRequireFits(entry, dtype: dtype)
        let availability = try await mcpAvailability(entry, dtype: dtype, forDownload: true)
        guard availability.needsGet else {
            let size = entry.sizesByDtype[dtype].flatMap { $0 > 0 ? " (\(MCPFormat.size($0)))" : nil } ?? ""
            return mcpText("\(entry.name) is already downloaded at \(dtype)\(size). Nothing to do.")
        }
        if let running = await mcpGetJob(model: entry.id, dtype: dtype) {
            return mcpText("Already downloading: \(await running.statusLine()). Job \(running.id) — call \(MCPNames.prefix)wait(job:\"\(running.id)\").")
        }
        if let answer = try await mcpGetGate(availability, body: body, verb: .download) { return answer }
        // The download puts the model on the user's table (it holds the models with files on this Mac). A model Add never
        // recorded joins it at the precision fetched; one the user added keeps the precision the user chose.
        let work: @Sendable (MCPJob) async throws -> CallTool.Result = { [self] job in
            try await self.mcpAcquire(job, availability)
            return mcpText(await job.downloadSummary() ?? "Downloaded \(entry.name).")
        }
        let job = try await startMCPJob(
            kind: .download, entry: entry, operation: nil, dtype: dtype, requester: requester,
            onCancel: { [self] job in await self.mcpJobCancelled(job, model: entry.id, dtype: dtype) },
            work: work)
        return try await mcpStarted(job, availability: availability, verb: .download, waitSeconds: 0, json: false, notify: notify)
    }

    func mcpLoad(_ body: JSONObject, requester: String, notify: MCPNotify?) async throws -> CallTool.Result {
        let entry = try await mcpEntry(mcpModelID(body)), dtype = try mcpDtype(body, entry: entry)
        try mcpRequireApproved(entry, verb: "Not loaded")
        try mcpRequireFits(entry, dtype: dtype)
        let availability = try await mcpAvailability(entry, dtype: dtype)
        if availability.needsGet, let answer = try await mcpGetGate(availability, body: body, verb: .load) { return answer }
        let fast = body["fast"]?.bool, service = service, environment = mcpEnvironment
        let work: @Sendable (MCPJob) async throws -> CallTool.Result = { [self] job in
            if availability.needsGet { try await self.mcpAcquire(job, availability) }
            await job.enter(.loading)
            let started = environment.now()
            try await service.load(id: entry.id, precision: dtype, fast: fast)
            let footprint = await service.memory().models[entry.id]?.footprintMB
            var text = "Loaded \(entry.name) at \(dtype)"
            if let footprint, footprint > 0 { text += " — \(MCPFormat.memory(decimalGB: footprint / 1000)) in memory" }
            text += ", took \(MCPFormat.elapsed(environment.now().timeIntervalSince(started)))."
            if let summary = await job.downloadSummary() { text = summary + "\n" + text }
            return mcpText(text)
        }
        let job = try await startMCPJob(
            kind: .load, entry: entry, operation: nil, dtype: dtype, requester: requester,
            onCancel: { [self] job in await self.mcpJobCancelled(job, model: entry.id, dtype: dtype) },
            work: work)
        if availability.needsGet {
            return try await mcpStarted(job, availability: availability, verb: .load, waitSeconds: 0, json: false, notify: notify)
        }
        return try await mcpWaitResult(job, seconds: 25, notify: notify, cancelJob: true, json: false, again: false)
    }

    func mcpRunTool(_ body: JSONObject, requester: String, notify: MCPNotify?) async throws -> CallTool.Result {
        let entry = try await mcpEntry(mcpModelID(body))
        let hint = "\(MCPNames.prefix)model(model:\"\(entry.id)\") lists the fields"
        guard let input = body["input"]?.object else { throw MCPFailure(.invalidInput, "Not run: input must be an object of the operation's fields; \(hint).") }
        let operations = await service.operations(id: entry.id)
        guard !operations.isEmpty else { throw MCPFailure(.unsupported, "Not run: \(entry.name) declares no operations.") }
        let name = body["operation"]?.string ?? RunAdmission.primaryOperation(operations, preferred: nil)
        guard let operation = operations.first(where: { $0.name == name }) else {
            throw MCPFailure(
                .unsupported,
                "Not run: \(entry.name) has no operation \"\(name)\". It offers: \(operations.map(\.name).joined(separator: ", ")). \(MCPNames.prefix)model(model:\"\(entry.id)\") shows their inputs."
            )
        }
        guard input["out_dir"] == nil else { throw MCPFailure(.invalidInput, "Not run: out_dir is managed by Arsenal; the result's folder is printed with it.") }
        let environment = mcpEnvironment
        // Ordered text members (a decision state, a template) are not staged: their "path" keys are not files.
        let verbatimNames = OrderedJSONMembers.verbatimNames(operation.inputSchema)
        let verbatim = input.filter { verbatimNames.contains($0.key) }
        let staged = input.filter { verbatim[$0.key] == nil }
        _ = try await mcpPrepareInputs(.object(staged), environment: environment, importing: false)
        do {
            // An ordered member (OrderedJSONMembers) may arrive as its JSON text; the service forwards that text exactly.
            try OperationSchemaValidator.validate(.object(try OrderedJSONMembers.split(input, schema: operation.inputSchema).request), schema: .object(operation.inputSchema))
        } catch {
            let message = (error as? WorkerFailure)?.message ?? (error as? HTTPFailure)?.message ?? error.localizedDescription
            throw MCPFailure(.invalidInput, "Not run: the input doesn't match \(operation.name): \(message.trimmingCharacters(in: CharacterSet(charactersIn: ". "))). \(hint).")
        }
        let dtype = entry.row.selectedPrecision
        try mcpRequireApproved(entry, verb: "Not run")
        try mcpRequireFits(entry, dtype: dtype)
        let availability = try await mcpAvailability(entry, dtype: dtype)
        if availability.needsGet, let answer = try await mcpGetGate(availability, body: body, verb: .run) { return answer }
        let (prepared, copies) = try await mcpPrepareInputs(.object(staged), environment: environment, importing: true)
        try mcpRequireCopyDisk(copies, environment: environment)
        let imported = (prepared.object ?? staged).merging(verbatim) { $1 }, support = support
        let play = body["play"]?.bool ?? (operation.name == "speak")
        let directory = try newJob()
        let pending = entry.row.revision?.pending != nil && availability.state == .loaded
        let work: @Sendable (MCPJob) async throws -> CallTool.Result = { [self] job in
            await job.setDirectory(directory)
            do {
                if !copies.isEmpty { try await APIRouter.mcpCopyInputs(copies, job: job) }
                if operation.schemaVersion == 3 {
                    do { try OperationArtifacts.input(.object(imported), schema: operation.inputSchema, support: support) } catch {
                        throw MCPFailure(.invalidInput, "Not run: \((error as? WorkerFailure)?.message ?? error.localizedDescription)")
                    }
                }
                if availability.needsGet { try await self.mcpAcquire(job, availability) }
                let rendered = try await self.mcpRun(entry: entry, operation: operation, input: imported, requester: requester, play: play, job: job, directory: directory)
                await job.setOutput(rendered.json)
                var text = rendered.text
                if let summary = await job.downloadSummary() { text = summary + "\n" + text }
                if pending { text += "\nA newer revision of \(entry.name) is ready; Arsenal swaps it in by itself when the model is idle." }
                return mcpText(text, extra: rendered.extra)
            } catch {
                await self.removeTransientJob(directory); throw error
            }
        }
        let job = try await startMCPJob(
            kind: .run, entry: entry, operation: operation.name, dtype: dtype, requester: requester,
            onCancel: { [self] job in await self.mcpJobCancelled(job, model: entry.id, dtype: dtype) }, work: work)
        let waitSeconds = min(45, max(0, body["wait_seconds"]?.number ?? 25)), json = body["format"]?.string == "json"
        if availability.needsGet {
            return try await mcpStarted(job, availability: availability, verb: .run, waitSeconds: waitSeconds, json: json, notify: notify)
        }
        return try await mcpWaitResult(job, seconds: waitSeconds, notify: notify, cancelJob: true, json: json, again: false)
    }

    /// The answer once a download has started: its size and ETA within `confirmSeconds`, never waiting for the whole Get.
    /// A Get that finishes inside that window (a local clone) continues into the run's normal wait.
    func mcpStarted(
        _ job: MCPJob, availability: MCPAvailability, verb: MCPConsentVerb, waitSeconds: Double, json: Bool, notify: MCPNotify?
    ) async throws -> CallTool.Result {
        let environment = mcpEnvironment, started = environment.now()
        try await withTaskCancellationHandler {
            while await !job.downloadMeasured, environment.now().timeIntervalSince(started) < environment.confirmSeconds {
                await job.waitForEnd(seconds: min(0.25, environment.confirmSeconds))
            }
            try Task.checkCancellation()
        } onCancel: {
            Task { await job.cancel() }
        }
        let phase = await job.phase
        if await job.status != .running || (phase != .downloading && phase != .deriving) {
            let left = max(0, waitSeconds - environment.now().timeIntervalSince(started))
            return try await mcpWaitResult(job, seconds: left, notify: notify, cancelJob: true, json: json, again: false)
        }
        let entry = availability.entry
        let total = await job.totalBytes ?? (availability.networkBytes ?? 0) + availability.localBytes
        var line: String
        if let rate = await job.rates.rate(at: environment.now()), rate > 0 {
            let left = Double(max(0, total - (await job.bytes))) / rate
            line = "Downloading \(entry.name): \(MCPFormat.size(total)), \(MCPFormat.remaining(left)) at \(MCPFormat.rate(rate))."
        } else if let rate = mcpRates.median(host: mcpHost(entry)), rate > 0, await job.bytes > 0 {
            line = "Downloading \(entry.name): \(MCPFormat.size(total)), \(MCPFormat.remaining(Double(total) / rate)) at this Mac's last measured \(MCPFormat.rate(rate))."
        } else if availability.networkBytes == 0 {
            line = "Preparing \(entry.name) from files already on this Mac" + (total > 0 ? " (\(MCPFormat.size(total)))." : ".")
        } else {
            line = "Download of \(entry.name) (\(MCPFormat.size(total))) is starting; speed not measured yet."
        }
        switch verb {
        case .run: line += " The \(job.operation ?? "run") runs as soon as it finishes."
        case .load: line += " It loads as soon as it finishes."
        case .download: break
        }
        let receive = verb == .run ? "to receive the result" : verb == .load ? "to hear when it is loaded" : "to hear when it finishes"
        let next = "Job \(job.id). Tell the user, then call \(MCPNames.prefix)wait(job:\"\(job.id)\") \(receive)."
        if json { return try mcpJSON(await mcpJobObject(job).merging(["message": .string(line), "next": .string(next)]) { _, b in b }) }
        return mcpText(line + "\n" + next)
    }

    /// Blocks until the job ends or `seconds` pass, sending progress when the client gave a token: at most every
    /// `tickSeconds` on a change, at least every `heartbeatSeconds`, `progress` strictly increasing (heartbeats omit `total`).
    func mcpWaitResult(
        _ job: MCPJob, seconds: Double, notify: MCPNotify?, cancelJob: Bool, json: Bool, again: Bool
    ) async throws -> CallTool.Result {
        let environment = mcpEnvironment
        return try await withTaskCancellationHandler {
            let deadline = environment.now().addingTimeInterval(seconds)
            var lastSent: Date?, lastRevision = -1, lastProgress = 0.0
            while true {
                if await job.status != .running { return await mcpFinal(job, json: json) }
                let now = environment.now()
                if now >= deadline { break }
                if let notify {
                    let revision = await job.revision
                    let due =
                        lastSent.map {
                            now.timeIntervalSince($0) >= environment.heartbeatSeconds || (revision != lastRevision && now.timeIntervalSince($0) >= environment.tickSeconds)
                        }
                        ?? true
                    if due {
                        let figure = await job.figure()
                        var value = lastProgress + 1, total: Double?
                        if let figure, figure.value >= value { value = figure.value; total = figure.total }
                        lastProgress = value; lastSent = now; lastRevision = revision
                        await notify(value, total, await job.statusLine())
                    }
                }
                try Task.checkCancellation()
                await job.waitForEnd(seconds: min(deadline.timeIntervalSince(now), environment.tickSeconds))
            }
            try Task.checkCancellation()
            let status = await job.statusLine()
            if json { return try mcpJSON(await mcpJobObject(job).merging(["message": .string(status)]) { _, b in b }) }
            if again { return mcpText("Still " + status.prefix(1).lowercased() + status.dropFirst() + ". Call \(MCPNames.prefix)wait(job:\"\(job.id)\") again.") }
            return mcpText("\(status). Job \(job.id) — call \(MCPNames.prefix)wait(job:\"\(job.id)\").")
        } onCancel: {
            if cancelJob { Task { await job.cancel() } }
        }
    }
    /// A finished job's answer: its result, or a cancelled line.
    func mcpFinal(_ job: MCPJob, json: Bool) async -> CallTool.Result {
        let status = await job.status
        if status == .cancelled || status == .interrupted {
            let downloaded = await job.hasDownloaded, phase = await job.phase
            let partial = downloaded && [MCPJob.Phase.downloading, .deriving, .queued].contains(phase) ? " Partial download kept; it resumes next time." : ""
            let how = status == .cancelled ? "cancelled" : "interrupted by an Arsenal restart"
            return MCPFailure(status == .cancelled ? .cancelled : .failed, "Job \(job.id) (\(await job.title())) was \(how).\(partial)").result
        }
        guard let result = await job.result else { return MCPFailure(.failed, "Job \(job.id) ended without a result.").result }
        if json, result.isError != true, let output = await job.output {
            var extra = result.content; if !extra.isEmpty { extra.removeFirst() }
            return (try? mcpJSON(output.merging(["job": .string(job.id), "model": .string(job.model)]) { a, _ in a }, extra: extra)) ?? result
        }
        return result
    }
    func mcpJobObject(_ job: MCPJob) async -> JSONObject {
        var object: JSONObject = [
            "job": .string(job.id), "kind": .string(job.kind.rawValue), "model": .string(job.model), "dtype": .string(job.dtype), "status": .string((await job.status).rawValue),
            "phase": .string((await job.phase).rawValue), "message": .string(await job.statusLine())
        ]
        if let operation = job.operation { object["operation"] = .string(operation) }
        if let total = await job.totalBytes { object["total_bytes"] = .number(Double(total)); object["bytes"] = .number(Double(await job.bytes)) }
        return object
    }

    func mcpWaitTool(_ body: JSONObject, notify: MCPNotify?) async throws -> CallTool.Result {
        guard let id = body["job"]?.string, !id.isEmpty else {
            throw MCPFailure(.invalidInput, "job must be a job id from \(MCPNames.prefix)run, \(MCPNames.prefix)load or \(MCPNames.prefix)download.")
        }
        let json = body["format"]?.string == "json"
        if let job = mcpJobs[id] {
            // Without a progress token nothing keeps the connection busy, so the stdio bridge's 60 s request timeout would
            // fire first: such waits are capped at the default 50 s.
            let seconds = min(notify == nil ? mcpEnvironment.waitSeconds : 3600, max(0, body["timeout_s"]?.number ?? mcpEnvironment.waitSeconds))
            return try await mcpWaitResult(job, seconds: seconds, notify: notify, cancelJob: false, json: json, again: true)
        }
        guard let record = mcpJobRecords[id] else { throw MCPFailure(.notFound, "No job \"\(id)\". \(MCPNames.prefix)status lists running and recent jobs.") }
        let title = record.kind == .run ? "\(record.operation ?? "run") with \(record.name)" : "\(record.kind.rawValue) of \(record.name)"
        switch record.status {
        case .interrupted, .running:
            throw MCPFailure(.failed, "Job \(id) (\(title)) was interrupted by an Arsenal restart. Call the same tool again; a download resumes where it stopped.")
        case .cancelled: throw MCPFailure(.cancelled, "Job \(id) (\(title)) was cancelled.")
        case .failed: return mcpText(record.line ?? "[failed] Job \(id) (\(title)) failed.", isError: true)
        case .completed:
            if let path = record.output, let text = try? String(contentsOfFile: path, encoding: .utf8) { return mcpText("Job \(id) (\(title)) finished.\n" + text) }
            return mcpText("Job \(id) (\(title)) finished: \(record.line ?? "done").")
        }
    }

    func mcpCancel(_ body: JSONObject) async throws -> CallTool.Result {
        if let id = body["job"]?.string, !id.isEmpty {
            if let job = mcpJobs[id] {
                let status = await job.status
                guard status == .running else { return mcpText("Job \(id) already ended (\(status.rawValue)); nothing to cancel.") }
                let downloaded = await job.hasDownloaded, phase = await job.phase
                let downloading = downloaded && [MCPJob.Phase.downloading, .deriving].contains(phase)
                await job.cancel(); await persist(job)
                return mcpText("Cancelled job \(id) (\(await job.title())).\(downloading ? " Partial download kept; it resumes next time." : "")")
            }
            if let record = mcpJobRecords[id] { return mcpText("Job \(id) already ended (\(record.status.rawValue)); nothing to cancel.") }
            throw MCPFailure(.notFound, "No job \"\(id)\". \(MCPNames.prefix)status lists running jobs.")
        }
        guard body["model"]?.string != nil else { throw MCPFailure(.invalidInput, "Pass job (a job id) or model (to stop its download).") }
        let entry = try await mcpEntry(mcpModelID(body))
        let row = await anyRow(entry.id)
        let downloading = row?.states.values.contains { $0.state == .downloading || $0.state == .deriving } == true || mcpGets.keys.contains { $0.hasPrefix(entry.id + "/") }
        guard downloading else { return mcpText("No download of \(entry.name) is running; nothing changed.") }
        for job in mcpJobs.values where job.model == entry.id {
            if await job.status == .running, await job.hasDownloaded, [MCPJob.Phase.downloading, .deriving].contains(await job.phase) { await job.cancel(); await persist(job) }
        }
        for key in mcpGets.keys where key.hasPrefix(entry.id + "/") { mcpGets.removeValue(forKey: key)?.task.cancel() }
        await service.cancelGet(id: entry.id)
        return mcpText("Stopped the download of \(entry.name). Partial download kept; it resumes next time.")
    }
}
