// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import LanguageWorker

struct OutputControlTests {
    @Test func leadingRoleMarkersAreRemovedFromFakeDecoderOutput() {
        // Index-Translate's live int4 German output began with a literal <assistant> tag before the translation.
        #expect(OutputControl.strip("<assistant>\n\n\n\nDer Rechnung ist am 12. November fällig.") == "Der Rechnung ist am 12. November fällig.")
        #expect(OutputControl.strip("<|im_start|>assistant\nHallo Welt<|im_end|>") == "Hallo Welt")
        #expect(OutputControl.strip("<start_of_turn>model\nHallo<end_of_turn>\n") == "Hallo")
        #expect(OutputControl.strip("</assistant>Hallo") == "Hallo")
    }
    @Test func taggedThinkingBlocksAreRemovedWhereverTheyOccur() {
        #expect(OutputControl.strip("<think>\nplan the verb\n</think>\n\nPrzeciągnij przycisk.") == "Przeciągnij przycisk.")
        #expect(OutputControl.strip("Halten Sie <think>x</think>die Taste.") == "Halten Sie die Taste.")
    }
    @Test func controlTokensAreTheSpecialTokensTheTemplateWrites() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let tokenizer: [String: Any] = [
            "added_tokens": [
                ["content": "<|im_end|>", "special": true], ["content": "<fcel>", "special": true],
                ["content": "<image>", "special": false], ["content": "<|image_pad|>", "special": true]
            ]
        ]
        try JSONSerialization.data(withJSONObject: tokenizer).write(to: folder.appendingPathComponent("tokenizer.json"))
        let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        let controls = OutputControl.controlTokens(directory: folder, template: template)
        #expect(controls == ["<|im_end|>"])
        // The template never writes <fcel> (a table cell token) or <|image_pad|>, so they are not turn markers.
        #expect(OutputControl.strip("<|im_start|>assistant\nDer Text<fcel><|im_end|>", controlTokens: controls) == "Der Text<fcel>")
        #expect(OutputControl.strip("<|im_end|>Ciao<|im_end|>", controlTokens: ["<|im_end|>"]) == "Ciao")
        #expect(OutputControl.controlTokens(directory: folder, template: nil).isEmpty)
    }
    @Test func markersInsideTheTextAndReasoningProseStayAsWritten() {
        // A marker in the middle is content, not a turn boundary.
        #expect(OutputControl.strip("Use the <assistant> tag here.") == "Use the <assistant> tag here.")
        // Free-form reasoning is not detected here: only markers and tagged blocks are removed.
        let prose = "Okay, let's tackle this translation. The user wants the text translated into German."
        #expect(OutputControl.strip(prose) == prose)
        // GroundGUI's tool-call block is the parser's input, so it must survive.
        let call = "<tool_call>{\"name\":\"computer_use\"}</tool_call>"
        #expect(OutputControl.strip(call) == call)
    }
}
