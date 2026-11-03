// SPDX-License-Identifier: MIT
import SwiftUI

/// The runner choices stay separate from the model's metadata, even at the browser's minimum width.
struct BrowserRunnerList: View {
    let detail: ModelDetail
    let palette: ArsenalPalette
    let choose: (String) -> Void

    /// What the list draws: the only runner's figures, or a card per runner (with radios when more than one can run).
    enum Layout: Equatable { case figures, cards(radios: Bool) }
    static func layout(_ detail: ModelDetail) -> Layout { detail.runners.count > 1 ? .cards(radios: detail.choosable) : .figures }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if detail.runners.count > 1 {
                // Several runners: a card each, chosen by its radio when more than one can run here.
                ForEach(detail.runners) { runner in
                    BrowserRunnerRow(runner: runner, palette: palette, choosable: detail.choosable, choose: choose)
                }
            } else if let only = detail.runners.first {
                // One runner is no choice: its figures alone, without a card, a name or a radio.
                BrowserRunnerFigures(runner: only, palette: palette)
            }
            if let note = detail.note {
                Text(note).font(.system(size: 11.5, weight: .medium)).foregroundStyle(palette.amber.swiftUI)
            }
        }
    }
}

/// The only runner's figures: the stat lines and why it cannot run here, in the detail's key/value style.
struct BrowserRunnerFigures: View {
    let runner: ModelDetail.Runner
    let palette: ArsenalPalette
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Which precision the figures are for: the row's own figure can be another one's (a dtype filter).
            RunnerPrecisionChip(runner: runner, palette: palette)
            RunnerStatLines(runner: runner, palette: palette)
            if let reason = runner.reason { Text(reason).font(.system(size: 11.5)).foregroundStyle(palette.amber.swiftUI) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityIdentifier("browser-runner-figures")
    }
}

/// Speed, peak memory and the kind's figures for the runner at the chosen precision.
struct RunnerStatLines: View {
    let runner: ModelDetail.Runner
    let palette: ArsenalPalette
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach(runner.stats) { stat in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Text(stat.title).foregroundStyle(palette.t2.swiftUI)
                    Spacer(minLength: 0)
                    if stat.missing {
                        ModelTableRow.triangle(size: 11).accessibilityLabel("Not available")
                    } else {
                        if let reference = stat.reference {
                            ModelTableRow.triangle(size: 9.5)
                            Text(reference).foregroundStyle(palette.t2.swiftUI)
                        } else if let delta = stat.delta {
                            Text(delta.text).foregroundStyle(ModelTable.tone(delta.tone, hot: false, palette))
                        }
                        Text(stat.value).foregroundStyle(palette.t1.swiftUI).monospacedDigit()
                    }
                }
                .font(.system(size: 11.5))
                .help(stat.help)
            }
        }
    }
}

struct BrowserRunnerRow: View {
    let runner: ModelDetail.Runner
    let palette: ArsenalPalette
    /// More than one runner can run here: each card has its radio; otherwise the name alone.
    var choosable = true
    let choose: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                if choosable {
                    // SwiftUI applies its environment's enabled state to the hosted control after its update: say it here too.
                    BrowserRunnerChoice(runner: runner, palette: palette, choose: choose).frame(height: 20).fixedSize().disabled(!runner.eligible)
                } else {
                    (Text(runner.name).font(.system(size: 12.5, weight: .semibold)).foregroundStyle((runner.eligible ? palette.t1 : palette.t4).swiftUI)
                        + Text(runner.engine.isEmpty ? "" : "  " + runner.engine).font(.system(size: 12)).foregroundStyle(palette.t2.swiftUI))
                        .lineLimit(1).help(runner.reason ?? runner.engineHelp)
                        .accessibilityIdentifier("browser-runner-" + runner.name)
                }
                RunnerPrecisionChip(runner: runner, palette: palette)
                Spacer(minLength: 0)
            }
            // What the runner does differently, in words (the difference between runners is the point of the card).
            if !runner.features.isEmpty {
                Text(runner.features.map(\.title).joined(separator: " \u{00b7} ")).font(.system(size: 11.5)).foregroundStyle(palette.t2.swiftUI)
                    .accessibilityIdentifier("browser-runner-features-" + runner.name)
            }
            RunnerStatLines(runner: runner, palette: palette)
            if let reason = runner.reason { Text(reason).font(.system(size: 11.5)).foregroundStyle(palette.amber.swiftUI) }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((runner.selected ? palette.field : palette.field.withAlphaComponent(0.35)).swiftUI, in: RoundedRectangle(cornerRadius: 7))
        .overlay { RoundedRectangle(cornerRadius: 7).strokeBorder((runner.selected ? palette.accent.withAlphaComponent(0.4) : palette.hairline).swiftUI) }
    }
}

/// The precision a runner's figures are for, as the Models table draws it: gold when it is the table's chosen one, edged
/// in white when loaded, with int4's triangle when its quality check did not pass or has not run.
struct RunnerPrecisionChip: View {
    let runner: ModelDetail.Runner
    let palette: ArsenalPalette
    var body: some View {
        let label = runner.precisionLabel.isEmpty ? runner.precision : runner.precisionLabel
        if !runner.precision.isEmpty {
            HStack(spacing: 3) {
                Text(label).font(.system(size: 11, weight: .medium))
                if runner.precisionWarning != nil {
                    Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 9, weight: .semibold))
                }
            }
            .foregroundStyle(Color(nsColor: runner.precisionChosen ? ArsenalPalette.ink : palette.t1))
            .padding(.horizontal, 5).frame(height: 16)
            .background((runner.precisionChosen ? ArsenalPalette.gold : palette.field).swiftUI, in: RoundedRectangle(cornerRadius: 4))
            .overlay {
                if runner.precisionLoaded { RoundedRectangle(cornerRadius: 4).strokeBorder(Color.white, lineWidth: 1.5) }
            }
            .help(
                [runner.precisionWarning, runner.precisionLoaded ? "Loaded at \(label)" : "The figures below are for \(label)"]
                    .compactMap { $0 }.joined(separator: "\n")
            )
            .accessibilityElement(children: .combine)
            .accessibilityIdentifier("browser-runner-precision")
        }
    }
}
