// SPDX-License-Identifier: MIT
import Foundation

// Models-table figures for folders whose own kind declares none (FolderReferenceFigures): Arsenal's shipped
// quick checks and the folder's own quick-check records. Nothing is written into the folder, whose every byte is bound
// to its code approval.
extension RealModelsService {
    /// The shipped figures (Resources/figures), then the figures again of every row they can apply to: local folders and
    /// signed kit models, whose rows were built before the bundled resources were known.
    func loadFolderFigures(_ resources: URL) {
        folderFigures = FolderReferenceFigures.load(resources: resources)
        for model in allManifests where localModels[model.id] != nil || ModelStatKind(rawValue: model.kind) == nil { updateMetrics(model) }
    }
    /// A local folder's own quick-check records among its evidence files (none for a catalog model).
    func cacheFolderFigureRecords(_ model: ModelManifest, files: [String: Data], local: LocalModelFolder?) {
        guard let local else { folderFigureRecords[model.id] = nil; return }
        let kind = local.embeddedKind?.spec ?? kinds().first { $0.id == model.kind }
        // The folder's own records measured its default runner, whatever runner is chosen now.
        let runner = model.selection?.defaultRunner ?? model.runners.first?.name ?? ""
        folderFigureRecords[model.id] = FolderReferenceFigures.records(files, model: model, runner: runner, kind: kind)
    }
    /// A row of a kind Arsenal has no stat kind for (a local folder's, or a signed kit model's): its figures by
    /// precision, for the chosen runner. A signed row has only the shipped figures, bound to its id, pins and runner.
    func shownFolderFigures(_ model: ModelManifest, row: ModelRow) -> [String: FolderReferenceFigures.Entry] {
        guard ModelStatKind(rawValue: model.kind) == nil else { return [:] }
        return folderFigures.figures(model, runner: row.runner, records: folderFigureRecords[model.id] ?? [])
    }
}
