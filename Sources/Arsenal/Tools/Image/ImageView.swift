// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// Make an Image, composed from the tool kit: the band (the model's pop-up of every image model in the table, its image
// operations as pills, Show in Finder), the live figures with their reference, the options the operation declares with
// Generate or Stop at their foot, and on the right the image with Copy and Show in Finder, this window's earlier
// images in the strip under it.

struct ImageView: View {
    @ObservedObject var session: ImageSession
    static let size = ToolLayout.size, minSize = ToolLayout.minSize

    var body: some View {
        // The reference row only for a model that has reference figures, so the band is one row like its siblings'.
        ToolFrame(figures: session.figures(), model: session.model?.model, reference: session.referenceLine, showsReference: session.hasReference) {
            SessionBand(session: session)
        } controls: {
            OptionsColumn(session: session, run: RunButton(session: session, title: session.runVerb, id: "generate"))
        } result: {
            ImageResultView(session: session)
        }
    }
}

/// The shown image with its actions; this window's images in the strip under the panel.
struct ImageResultView: View {
    @ObservedObject var session: ImageSession

    var body: some View {
        VStack(spacing: 12) {
            ResultPanel(session: session, symbol: ToolKind.image.symbol, empty: session.current == nil) {
                if let current = session.current {
                    VStack(spacing: 0) {
                        ResultActions(what: "image", enabled: !session.running, copy: { session.copy(current.url) }, save: { session.saveAs(current.url) })
                        FittedImage(
                            url: current.url, size: current.width > 0 && current.height > 0 ? PixelSize(width: current.width, height: current.height) : nil,
                            draggable: true, label: current.prompt, help: current.prompt
                        ).padding(.horizontal, 16).padding(.bottom, 16)
                    }
                }
            }
            ImageStrip(
                items: session.results.map { .init(id: $0.id, url: $0.url, label: $0.prompt, controlID: "result-\($0.url.lastPathComponent)") },
                current: session.current?.id, select: { session.shown = $0 }
            ).historySlot(!session.results.isEmpty)
        }
    }
}
