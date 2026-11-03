// SPDX-License-Identifier: MIT
import Foundation
import AppKit
import Testing

@testable import Arsenal

/// The Docs popover draws the model README's Markdown as native blocks.
@Suite struct ModelReadmeTests {
    @Test func readmeBlocksParseInOrder() {
        let text = """
            <!-- SPDX-License-Identifier: MIT -->
            # BiRefNet

            Removes an image's background
            in one pass.

            ## Inputs and outputs

            | Field | Type |
            |---|---|
            | `image` | file |

            - first use
              continues here
            - second use

            ```sh
            curl -s "$ARSENAL_URL"
            ```
            """
        #expect(
            MarkdownBlock.parse(text) == [
                .heading(1, "BiRefNet"), .paragraph("Removes an image's background in one pass."), .heading(2, "Inputs and outputs"),
                .table([["Field", "Type"], ["`image`", "file"]]), .bullets(["first use continues here", "second use"]), .code("curl -s \"$ARSENAL_URL\"")
            ])
    }

    @Test func numberedListsKeepTheirNumbersAndTablesKeepAnEmptyLastCell() {
        let text = """
            1. Load the audio.
            2. Find speech.
            - a bullet after
            | a | b |
            |---|---|
            | x | |
            """
        #expect(
            MarkdownBlock.parse(text) == [
                .numbered(["1. Load the audio.", "2. Find speech."]), .bullets(["a bullet after"]), .table([["a", "b"], ["x", ""]])
            ])
    }

    @MainActor @Test func docsButtonKeepsItsLabelPressedAndOpensOnePopoverPerClick() async throws {
        let view = ModelReadmeButton.ReadmeButton()
        let title = NSAttributedString(string: "Docs")
        view.attributedTitle = title
        #expect(view.attributedAlternateTitle.string == "Docs")
        let calls = Calls()
        view.load = {
            calls.count += 1
            try? await Task.sleep(nanoseconds: 50_000_000)
            return "# Title"
        }
        view.toggle()
        view.toggle()
        try await Task.sleep(nanoseconds: 200_000_000)
        #expect(calls.count == 1)
        // Without a window nothing is shown, so no popover floats unanchored.
        #expect(!view.isShowingReadme)
    }

    @Test func everyShippedReadmeParsesToHeadingsAndText() throws {
        let models = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("models")
        let ids = try FileManager.default.contentsOfDirectory(atPath: models.path).filter { !$0.hasPrefix(".") }
        #expect(ids.count >= 37)
        for id in ids {
            let blocks = MarkdownBlock.parse(try String(contentsOf: models.appendingPathComponent("\(id)/README.md"), encoding: .utf8))
            #expect(blocks.first.map { if case .heading(1, _) = $0 { true } else { false } } == true, "\(id)")
            #expect(blocks.count > 5, "\(id)")
        }
    }
}

private final class Calls: @unchecked Sendable { var count = 0 }
