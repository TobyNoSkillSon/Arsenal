// SPDX-License-Identifier: MIT
import Foundation

/// The shipped folder, or the withdrawn one in retired-models/. Non-open models are kept there, not shipped, and their
/// fixtures and contracts stay covered.
func shippedOrRetiredFolder(_ repo: URL, _ id: String) -> URL {
    let shipped = repo.appendingPathComponent("models/" + id)
    return FileManager.default.fileExists(atPath: shipped.path) ? shipped : repo.appendingPathComponent("retired-models/" + id)
}
