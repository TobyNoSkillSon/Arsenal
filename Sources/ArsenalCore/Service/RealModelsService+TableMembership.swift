// SPDX-License-Identifier: MIT
import Foundation

/// Putting a model on the user's table. Remove stays beside unload in RealModelsService.
///
/// The table shows exactly the models with weights on this Mac or a Get under way (`ModelRow.holdsWeightsOrGet`).
/// `config.table` keeps the order models joined in and that Add recorded them (Show in menu off, recommended precision);
/// an entry without weights stays in Get more models until its Get starts.
extension RealModelsService {
    /// Whether `id` is on the Models table now.
    /// A withdrawn catalog model is kept only while its weights may be on disk; a local folder's problem row is no Get, and
    /// a broken local folder stays where the user put it.
    func onTable(_ id: String) -> Bool {
        if gets[id] != nil || droppedCatalogModels[id] != nil { return true }
        if localProblemIDs.contains(id) { return false }
        if brokenLocalFolders[id] != nil { return config.table.contains(id) || residents[id] != nil }
        return modelRows[id]?.holdsWeightsOrGet == true
    }

    /// A Get of a model Add never recorded (an agent's, through MCP or the API) records it as Add would, keeping the
    /// precision being fetched, so it joins the table with Show in menu off.
    func joinTable(_ id: String, precision: String) async throws {
        guard !config.table.contains(id) else { return }
        try await add(id: id)
        config.selections[id]?.precision = precision
        try persist()
    }

    public func add(id: String) async throws {
        try admitNewWork(); try admitWork(); await refreshLocalModels(); let model = try manifest(id)
        guard !config.table.contains(id) else { return }
        let previousSelection = config.selections[id]
        var preference = previousSelection ?? .init(precision: model.defaultPrecision)
        if previousSelection?.explicitSelection != true {
            preference.precision = recommendedDtype(model); preference.fast = true; preference.explicitSelection = true
        }
        // A model added to the table stays out of the main menu until the user or an agent turns Show in menu on.
        preference.toolsShown = false
        config.selections[id] = preference
        try enqueueMeasurements(model)
        let previous = config.table; config.table.append(id)
        do { try persist() } catch { config.table = previous; config.selections[id] = previousSelection; throw error }
        modelRows[id]?.toolsShown = false; catalogEntryCache[id] = nil
        modelRows[id]?.selectedPrecision = servablePrecision(model, config.selections[id]?.precision ?? model.defaultPrecision)
        let fastAvailable = modelRows[id]?.fastAvailable == true
        modelRows[id]?.fast = fastAvailable; updatePending(id); changed()
    }
}
