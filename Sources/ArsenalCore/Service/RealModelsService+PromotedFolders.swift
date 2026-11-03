// SPDX-License-Identifier: MIT
import Foundation

// A local folder whose id the signed catalog now lists (a local model promoted to the catalog, such as the kit models
// bundled into the native worker) is superseded: ignored, never run, left on disk, with no error row. The signed row the
// catalog built takes its place and keeps its row, metadata and settings. The weights both share stay put.
extension RealModelsService {
    /// A local folder no longer discovered: its row goes, unless the signed catalog now lists its id.
    func forgetVanishedLocalFolder(_ id: String) {
        if catalog.models.contains(where: { $0.id == id }) { forgetPromotedLocalFolders([id]); return }
        modelRows[id] = nil; metadata[id] = nil; modelSettings.declarations[id] = nil
    }
    /// Forget the folders' code approvals and revisions: signed code needs no approval, and no folder revision may come
    /// back under the signed id.
    func forgetPromotedLocalFolders(_ ids: [String]) {
        var changed = false
        for id in ids {
            previousRevisions[id] = nil; dormantRevisions[id] = nil; rejectedRevisionDigests[id] = nil
            modelRows[id]?.approval = nil; modelRows[id]?.needsCodeConsent = false
            updateRevisionRow(id)
            guard config.localCodeApprovals[id] != nil || config.localCodeConsent[id] != nil || config.rejectedRevisions[id] != nil else { continue }
            config.localCodeApprovals[id] = nil; config.localCodeConsent[id] = nil; config.rejectedRevisions[id] = nil
            changed = true
        }
        guard changed else { return }
        do { try persist() } catch { emit(.error(id: nil, message: "Promoted folders' approvals could not be cleared: \(error.localizedDescription)")) }
    }
}
