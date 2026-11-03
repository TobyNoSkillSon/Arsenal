// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Combine

/// The one question a model's Delete asks: the size that will be freed and a single Delete button. No
/// keep-weights choice on this path; the service keeps its own options for other callers.
///
/// A precision line's trash asks through the same hook (`TableController.confirmDeleteWeights`, which the menu shows
/// with its own alert presenter): while `copy` is set, `alert` asks about that one copy instead of the whole model.
enum DeleteConfirm {
    /// One precision's copy: its label ("16-bit"), whether Get can download it again, and its precision ("bf16").
    struct Copy: Equatable, Sendable {
        var label: String
        var downloadable: Bool
        var dtype: String = ""
    }
    /// The copy the question in flight is about; nil for the model's whole Delete.
    @TaskLocal static var copy: Copy?

    static func alert(_ row: ModelRow, bytes: Int64) -> NSAlert {
        if let copy { return copyAlert(row, copy: copy, bytes: bytes) }
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Delete \(row.name)?"
        alert.informativeText =
            bytes > 0
            ? "Frees up to \(formatBytes(bytes)). Every precision and cache goes, and the model leaves the table."
            : "Every precision and cache goes, and the model leaves the table."
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Delete").hasDestructiveAction = true
        return alert
    }

    /// "Delete the 16-bit copy of Kokoro 82M? 164 MB is freed. It can be downloaded again."
    static func copyAlert(_ row: ModelRow, copy: Copy, bytes: Int64) -> NSAlert {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Delete the \(copy.label) copy of \(row.name)?"
        let again = copy.downloadable ? "It can be downloaded again." : "It is no longer offered, so it cannot be downloaded again."
        alert.informativeText = bytes > 0 ? "\(formatBytes(bytes)) is freed. \(again)" : again
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Delete").hasDestructiveAction = true
        return alert
    }
}

/// A question whose subject can go while it is up (the model removed over MCP, its weights deleted elsewhere) closes
/// itself: after each change it re-reads `stands`, and once that fails it calls `close` once (the menu ends the
/// alert's modal session as Cancel: `AppModal.cancelCurrent`).
@MainActor final class StaleQuestion {
    private var watch: AnyCancellable?
    init(_ changes: ObservableObjectPublisher, stands: @escaping @MainActor () -> Bool, close: @escaping @MainActor () -> Void) {
        watch = changes.sink { [weak self] in
            // `objectWillChange` fires before the new value is stored: read it on the next turn of the main queue.
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self, self.watch != nil, !stands() else { return }
                    self.end(); close()
                }
            }
        }
    }
    func end() { watch = nil }
}
