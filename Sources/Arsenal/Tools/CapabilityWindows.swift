// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// The capability windows by tool: each one's session and view, so ToolCenter opens any of them the same way. Adding a
// window is a session and a view in its own folder, a line here and its operations in ToolCatalog.

@MainActor enum CapabilityWindows {
    static func make(
        _ tool: ToolKind, models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, output: URL?, scratch: URL? = nil
    ) -> (CapabilitySession, AnyView) {
        switch tool {
        case .transcribe:
            let s = TranscribeSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(TranscribeView(session: s)))
        case .separate:
            let s = SeparateSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(SeparateView(session: s)))
        case .search:
            let s = SearchSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(SearchView(session: s)))
        case .translate:
            let s = TranslateSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(TranslateView(session: s)))
        case .document:
            let s = DocumentSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(DocumentView(session: s)))
        case .extract:
            let s = ExtractSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(ExtractView(session: s)))
        case .select:
            let s = SelectSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(SelectView(session: s)))
        case .find:
            let s = FindSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(FindView(session: s)))
        case .depth:
            let s = DepthSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(DepthView(session: s)))
        case .upscale:
            let s = UpscaleSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(UpscaleView(session: s)))
        case .smooth:
            let s = SmoothSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(SmoothView(session: s)))
        case .make3d:
            let s = Make3DSession(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: output, scratch: scratch)
            return (s, AnyView(Make3DView(session: s)))
        default:
            preconditionFailure("\(tool) is not a capability window")
        }
    }
}
