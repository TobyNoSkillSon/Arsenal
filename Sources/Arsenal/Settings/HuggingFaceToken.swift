// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore

/// Settings › Hugging Face Token…: one native alert with a secure field. Save writes Arsenal's Keychain item (replacing
/// any earlier token), Remove deletes it. The token is never logged, shown or sent anywhere but huggingface.co, and no
/// API or MCP route can set it. After a save, Gets waiting on a gated model retry at once.
@MainActor enum HuggingFaceToken {
    static let title = "Hugging Face Token\u{2026}"
    static let help = "For models whose terms you accept on Hugging Face. Kept in your Keychain."
    /// What the alert did.
    enum Outcome: Equatable { case saved, removed, cancelled, failed(String) }
    /// The Keychain, injectable for tests.
    struct Store {
        var write: (String) -> OSStatus = { HFTokenStore.write($0) }
        var delete: () -> OSStatus = { HFTokenStore.delete() }
        var hasToken: () -> Bool = { HFTokenStore.hasStoredToken }
    }

    /// Shows the alert and applies the answer.
    static func edit(store: Store = Store(), service: (any ModelsService)?) -> Outcome {
        let stored = store.hasToken()
        let alert = NSAlert()
        alert.messageText = "Hugging Face Token"
        alert.informativeText = stored ? "A token is saved. Paste a new one to replace it." : "Paste a read token from huggingface.co/settings/tokens."
        let field = NSSecureTextField(frame: NSRect(x: 0, y: 0, width: 300, height: 24))
        field.placeholderString = "hf_\u{2026}"
        alert.accessoryView = field
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")
        if stored { alert.addButton(withTitle: "Remove") }
        alert.window.initialFirstResponder = field
        NSApp.activate(ignoringOtherApps: true)
        let response = alert.runModal()
        let outcome = apply(response, text: field.stringValue, store: store)
        field.stringValue = ""
        switch outcome {
        case .saved: if let service { Task { await service.retryGatedGets() } }
        case .failed(let reason):
            let problem = NSAlert(); problem.messageText = "Token not saved"; problem.informativeText = reason; problem.runModal()
        case .removed, .cancelled: break
        }
        return outcome
    }

    /// The alert's answer applied to the store: first button saves, third removes, anything else changes nothing.
    static func apply(_ response: NSApplication.ModalResponse, text: String, store: Store) -> Outcome {
        switch response {
        case .alertFirstButtonReturn:
            let token = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !token.isEmpty else { return .cancelled }
            guard !token.contains(where: \.isWhitespace) else { return .failed("A token is one word with no spaces, like hf_\u{2026}") }
            let status = store.write(token)
            return status == errSecSuccess ? .saved : .failed("The Keychain refused it (\(status)).")
        case .alertThirdButtonReturn:
            let status = store.delete()
            return status == errSecSuccess ? .removed : .failed("The Keychain refused it (\(status)).")
        default: return .cancelled
        }
    }
}

/// Opens Hugging Face pages in the default browser, on the main thread.
enum HuggingFacePages {
    @Sendable static func open(_ pages: [URL]) {
        DispatchQueue.main.async { for page in pages where page.host == "huggingface.co" { NSWorkspace.shared.open(page) } }
    }
}
