// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// What an agent reads when a Get waits for a person to accept a gated model's terms on Hugging Face (MCP's
/// `[waiting_on_person]`, HTTP's `waiting_on_person`). The app has opened the pages and retries by itself; no route sets a token.
enum GatedModelText {
    static func line(
        name: String, page: String, hasToken: Bool, mode: String?, limit: Int = 10, tokenRejected: Bool = false, gaveUp: Bool = false
    ) -> String {
        let retries =
            gaveUp
            ? "Arsenal stopped retrying after \(limit) tries; once the user has done that, they click Get in Arsenal (or save the token there), which starts the download."
            : "Arsenal opened \(hasToken ? "that page" : "both pages") and retries every 30 s, up to \(limit) times."
        let manual = mode == "manual" ? " Its authors approve each request, which can take days." : ""
        let tail =
            gaveUp
            ? "Tell the user." : "Tell the user; once they accept, Arsenal's next retry downloads it (calling again meanwhile only reports this wait)."
        if hasToken {
            return "\(name) is gated on Hugging Face: the user must accept its terms at \(page), signed in to Hugging Face. \(retries)\(manual) "
                + tail
        }
        let missing = tokenRejected ? "Hugging Face refused Arsenal's token (expired or revoked)" : "Arsenal has no Hugging Face token"
        return "\(name) is gated on Hugging Face and \(missing): the user must create a read token at \(HFGateError.tokenPage.absoluteString), "
            + "save it in Arsenal (Settings \u{203a} Hugging Face Token\u{2026}; agents cannot set it), and accept the model's terms at \(page). \(retries)\(manual) "
            + tail
    }
    static func line(_ gate: HFGateError, name: String) -> String {
        line(name: name, page: gate.page.absoluteString, hasToken: gate.hasToken, mode: gate.mode, tokenRejected: gate.tokenRejected)
    }
    static func line(_ wait: GateWait, name: String) -> String {
        line(
            name: name, page: wait.page, hasToken: wait.hasToken, mode: wait.mode, limit: wait.limit, tokenRejected: wait.tokenRejected == true,
            gaveUp: wait.gaveUp)
    }
    static func details(_ wait: GateWait) -> JSONObject {
        var details: JSONObject = [
            "repo": .string(wait.repo), "page": .string(wait.page), "has_token": .bool(wait.hasToken), "tries": .number(Double(wait.tries)),
            "limit": .number(Double(wait.limit)), "retrying": .bool(!wait.gaveUp)
        ]
        if let mode = wait.mode { details["gate"] = .string(mode) }
        if wait.tokenRejected == true { details["token_rejected"] = .bool(true) }
        if !wait.hasToken { details["token_page"] = .string(HFGateError.tokenPage.absoluteString) }
        return details
    }
    static func details(_ gate: HFGateError) -> JSONObject {
        var details: JSONObject = [
            "repo": .string(gate.repo), "page": .string(gate.page.absoluteString), "has_token": .bool(gate.hasToken), "retry_seconds": .number(30)
        ]
        if let mode = gate.mode { details["gate"] = .string(mode) }
        if gate.tokenRejected { details["token_rejected"] = .bool(true) }
        if !gate.hasToken { details["token_page"] = .string(HFGateError.tokenPage.absoluteString) }
        return details
    }
}
