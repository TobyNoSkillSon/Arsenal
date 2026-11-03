// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

/// The browser already embeds AppKit controls in its hosted rows. A native radio exposes the same click and AX path.
struct BrowserRunnerChoice: NSViewRepresentable {
    let runner: ModelDetail.Runner
    let palette: ArsenalPalette
    let choose: (String) -> Void

    func makeNSView(context: Context) -> RunnerChoiceButton {
        let button = RunnerChoiceButton()
        button.setButtonType(.radio)
        button.target = button; button.action = #selector(RunnerChoiceButton.pick)
        button.setContentHuggingPriority(.defaultLow, for: .horizontal)
        button.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return button
    }
    func updateNSView(_ button: RunnerChoiceButton, context: Context) {
        let name = NSAttributedString(
            string: runner.name, attributes: [.font: NSFont.systemFont(ofSize: 12.5, weight: .semibold), .foregroundColor: runner.eligible ? palette.t1 : palette.t4])
        let title = NSMutableAttributedString(attributedString: name)
        if !runner.engine.isEmpty {
            title.append(NSAttributedString(string: "  " + runner.engine, attributes: [.font: NSFont.systemFont(ofSize: 12), .foregroundColor: palette.t2]))
        }
        button.attributedTitle = title
        button.lineBreakMode = .byTruncatingTail
        button.state = runner.selected ? .on : .off
        button.isEnabled = runner.eligible
        button.toolTip = runner.reason ?? runner.engineHelp
        button.onPick = { choose(runner.name) }
        button.setAccessibilityIdentifier("browser-runner-" + runner.name)
        button.setAccessibilityLabel("Use " + runner.name + " " + runner.engine + (runner.isDefault ? ", default; restores automatic selection" : ""))
    }
}

final class RunnerChoiceButton: NSButton {
    var onPick: () -> Void = {}
    @objc func pick() { onPick() }
}
