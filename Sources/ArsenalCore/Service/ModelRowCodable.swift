// SPDX-License-Identifier: MIT
import Foundation

extension ModelRow {
    private enum CodingKeys: String, CodingKey {
        case precisionDtypes, precisionNotes, loadMemoryNote, withdrawnTiers, recommendedPrecision, recommendationPending, recommendationReason, precisionStats,
            precisionQualityWarnings, revision,
            checks, purpose,
            tools,
            toolsShown, activity,
            formats,
            runnerOverride,
            defaultRunner, residentRunner, approval,
            settingsWarning, id, name,
            kind,
            params,
            licence, precisions, states,
            selectedPrecision, fast, runner, badge, metrics, peakRAMMB, keepHot, sizeBytes, summary, downloadBytes, verification, publisher, released, commercial, runnerLanguage,
            fastExact, quickMetrics, retentionNote, localCopyBytes, needsReload, local, contentDigest, hasFastKernels, kernels, needsCodeConsent, codeConsentRequired,
            fastAvailable, onTable
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try c.decode(String.self, forKey: .id),
            name: try c.decode(String.self, forKey: .name),
            kind: try c.decode(String.self, forKey: .kind),
            params: try c.decodeIfPresent(String.self, forKey: .params) ?? "",
            licence: try c.decodeIfPresent(String.self, forKey: .licence) ?? "",
            precisions: try c.decodeIfPresent([String].self, forKey: .precisions) ?? ["bf16"],
            states: try c.decodeIfPresent([String: PrecisionState].self, forKey: .states) ?? [:],
            selectedPrecision: try c.decodeIfPresent(String.self, forKey: .selectedPrecision) ?? "bf16",
            fast: try c.decodeIfPresent(Bool.self, forKey: .fast) ?? true,
            runner: try c.decodeIfPresent(String.self, forKey: .runner) ?? "native",
            badge: try c.decodeIfPresent(ModelBadge.self, forKey: .badge) ?? .unverified,
            metrics: try c.decodeIfPresent(JSONObject.self, forKey: .metrics) ?? [:],
            peakRAMMB: try c.decodeIfPresent(Double.self, forKey: .peakRAMMB),
            keepHot: try c.decodeIfPresent(Bool.self, forKey: .keepHot) ?? false,
            sizeBytes: try c.decodeIfPresent(Int64.self, forKey: .sizeBytes),
            summary: try c.decodeIfPresent(String.self, forKey: .summary) ?? "",
            downloadBytes: try c.decodeIfPresent([String: Int64].self, forKey: .downloadBytes) ?? [:],
            verification: try c.decodeIfPresent(String.self, forKey: .verification) ?? "Not tried on this Mac yet",
            publisher: try c.decodeIfPresent(String.self, forKey: .publisher) ?? "",
            released: try c.decodeIfPresent(Int.self, forKey: .released),
            commercial: try c.decodeIfPresent(Bool.self, forKey: .commercial) ?? true,
            runnerLanguage: try c.decodeIfPresent(String.self, forKey: .runnerLanguage) ?? "swift",
            fastExact: try c.decodeIfPresent(Bool.self, forKey: .fastExact) ?? true,
            quickMetrics: try c.decodeIfPresent([String: Bool].self, forKey: .quickMetrics) ?? [:],
            retentionNote: try c.decodeIfPresent(String.self, forKey: .retentionNote),
            localCopyBytes: try c.decodeIfPresent([String: Int64].self, forKey: .localCopyBytes) ?? [:]
        )
        loadMemoryNote = try c.decodeIfPresent(String.self, forKey: .loadMemoryNote)
        withdrawnTiers = try c.decodeIfPresent([String: Int64].self, forKey: .withdrawnTiers) ?? [:]
        onTable = try c.decodeIfPresent(Bool.self, forKey: .onTable) ?? true
        recommendedPrecision = try c.decodeIfPresent(String.self, forKey: .recommendedPrecision)
        recommendationPending = try c.decodeIfPresent(Bool.self, forKey: .recommendationPending) ?? true
        recommendationReason = try c.decodeIfPresent(String.self, forKey: .recommendationReason) ?? "Precision measurements are pending."
        precisionDtypes = try c.decodeIfPresent([String: String].self, forKey: .precisionDtypes) ?? [:]
        precisionNotes = try c.decodeIfPresent([String: String].self, forKey: .precisionNotes) ?? [:]
        precisionStats = try c.decodeIfPresent([String: PrecisionStatSet].self, forKey: .precisionStats) ?? [:]
        revision = try c.decodeIfPresent(ModelRevision.self, forKey: .revision)
        checks = try c.decodeIfPresent(ModelChecksStatus.self, forKey: .checks)
        purpose = try c.decodeIfPresent(String.self, forKey: .purpose) ?? ""
        tools = try c.decodeIfPresent([String].self, forKey: .tools) ?? []
        toolsShown = try c.decodeIfPresent(Bool.self, forKey: .toolsShown) ?? true
        activity = try c.decodeIfPresent(String.self, forKey: .activity) ?? tools.first ?? "default"
        formats = try c.decodeIfPresent([ModelFormat].self, forKey: .formats) ?? []
        runnerOverride = try c.decodeIfPresent(String.self, forKey: .runnerOverride)
        defaultRunner = try c.decodeIfPresent(String.self, forKey: .defaultRunner)
        residentRunner = try c.decodeIfPresent(String.self, forKey: .residentRunner)
        approval = try c.decodeIfPresent(CodeApproval.self, forKey: .approval)
        settingsWarning = try c.decodeIfPresent(String.self, forKey: .settingsWarning)
        needsReload = try c.decodeIfPresent(Bool.self, forKey: .needsReload) ?? false
        local = try c.decodeIfPresent(Bool.self, forKey: .local) ?? false
        contentDigest = try c.decodeIfPresent(String.self, forKey: .contentDigest)
        hasFastKernels = try c.decodeIfPresent(Bool.self, forKey: .hasFastKernels) ?? false
        kernels = try c.decodeIfPresent([JSONObject].self, forKey: .kernels) ?? []
        needsCodeConsent = try c.decodeIfPresent(Bool.self, forKey: .codeConsentRequired) ?? c.decodeIfPresent(Bool.self, forKey: .needsCodeConsent) ?? false
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(recommendedPrecision, forKey: .recommendedPrecision)
        try c.encode(recommendationPending, forKey: .recommendationPending)
        try c.encode(recommendationReason, forKey: .recommendationReason)
        try c.encode(precisionDtypes, forKey: .precisionDtypes)
        try c.encode(precisionNotes, forKey: .precisionNotes)
        try c.encode(precisionStats, forKey: .precisionStats)
        try c.encode(precisionQualityWarnings, forKey: .precisionQualityWarnings)
        try c.encodeIfPresent(revision, forKey: .revision)
        try c.encodeIfPresent(checks, forKey: .checks)
        try c.encode(purpose, forKey: .purpose)
        try c.encode(tools, forKey: .tools)
        try c.encode(toolsShown, forKey: .toolsShown)
        try c.encode(activity, forKey: .activity)
        try c.encode(formats, forKey: .formats)
        try c.encodeIfPresent(runnerOverride, forKey: .runnerOverride)
        try c.encodeIfPresent(defaultRunner, forKey: .defaultRunner)
        try c.encodeIfPresent(residentRunner, forKey: .residentRunner)
        try c.encodeIfPresent(approval, forKey: .approval)
        try c.encodeIfPresent(settingsWarning, forKey: .settingsWarning)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(kind, forKey: .kind)
        try c.encode(params, forKey: .params)
        try c.encode(licence, forKey: .licence)
        try c.encode(precisions, forKey: .precisions)
        try c.encode(states, forKey: .states)
        try c.encode(selectedPrecision, forKey: .selectedPrecision)
        try c.encode(fast, forKey: .fast)
        try c.encode(runner, forKey: .runner)
        try c.encode(badge, forKey: .badge)
        try c.encode(metrics, forKey: .metrics)
        try c.encodeIfPresent(peakRAMMB, forKey: .peakRAMMB)
        try c.encode(keepHot, forKey: .keepHot)
        try c.encodeIfPresent(sizeBytes, forKey: .sizeBytes)
        try c.encode(summary, forKey: .summary)
        try c.encode(downloadBytes, forKey: .downloadBytes)
        try c.encode(verification, forKey: .verification)
        try c.encode(publisher, forKey: .publisher)
        try c.encodeIfPresent(released, forKey: .released)
        try c.encode(commercial, forKey: .commercial)
        try c.encode(runnerLanguage, forKey: .runnerLanguage)
        try c.encode(fastExact, forKey: .fastExact)
        try c.encode(quickMetrics, forKey: .quickMetrics)
        try c.encodeIfPresent(retentionNote, forKey: .retentionNote)
        try c.encode(localCopyBytes, forKey: .localCopyBytes)
        try c.encodeIfPresent(loadMemoryNote, forKey: .loadMemoryNote)
        if !withdrawnTiers.isEmpty { try c.encode(withdrawnTiers, forKey: .withdrawnTiers) }
        if !onTable { try c.encode(onTable, forKey: .onTable) }
        try c.encode(needsReload, forKey: .needsReload)
        try c.encode(local, forKey: .local)
        try c.encodeIfPresent(contentDigest, forKey: .contentDigest)
        try c.encode(hasFastKernels, forKey: .hasFastKernels)
        try c.encode(kernels, forKey: .kernels)
        try c.encode(needsCodeConsent, forKey: .needsCodeConsent)
        try c.encode(codeConsentRequired, forKey: .codeConsentRequired)
        try c.encode(fastAvailable, forKey: .fastAvailable)
    }
}
