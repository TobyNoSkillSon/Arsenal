// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// The tool kit's frame, shared by every tool window: the band on the traffic lights' line (the model, as a pop-up where
// the window offers more than one, how it runs, the operations as pills, the window's actions), the live figures, then
// the options on the left with the run button at their foot and the result on the right.

enum ToolLayout {
    static let size = NSSize(width: 1040, height: 720), minSize = NSSize(width: 760, height: 520)
    static let controlsWidth: CGFloat = 300
    static let inset: CGFloat = 20
    /// The traffic lights end at 69 pt in a titled window on macOS 26 (measured in V3ToolTests); the name starts 14 after.
    static let bandLeading: CGFloat = 83
}

/// The lab figures (time, length, memory, references) are details: hidden until the band's Details button shows them,
/// one choice for every tool window, kept across launches.
enum ToolDetails {
    static let key = "ToolDetailsShown"
    static var shown: Bool {
        get { UserDefaults.standard.bool(forKey: key) }
        set { UserDefaults.standard.set(newValue, forKey: key) }
    }
}
/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
typealias ToolDetailsShown = SwiftUI.AppStorage<Bool>

/// A tool window: band, figures (with Details on), then options | result.
struct ToolFrame<Band: View, Controls: View, Result: View>: View {
    let figures: [Figure]?
    /// The model the window runs: its precision ends the Details strip, so it is not lost from the window.
    var model: ToolModel?
    var reference: ReferenceLine?
    var showsReference = false
    @ViewBuilder let band: Band
    @ViewBuilder let controls: Controls
    @ViewBuilder let result: Result
    @Environment(\.palette) private var p
    @ToolDetailsShown(ToolDetails.key) private var details = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            band
            if details, let figures {
                // Only an elapsed time still running ticks; otherwise the grid redraws on data alone.
                TelemetryGrid(figures: figures + (model.map { [$0.precisionFigure] } ?? []), reference: reference, showsReference: showsReference)
                    .padding(.horizontal, ToolLayout.inset).padding(.top, 6).padding(.bottom, 14)
            }
            Rectangle().fill(p.hairline.swiftUI).frame(height: 1)
            HStack(alignment: .top, spacing: 0) {
                controls.frame(width: ToolLayout.controlsWidth).padding(ToolLayout.inset)
                Rectangle().fill(p.hairline.swiftUI).frame(width: 1)
                result.padding(ToolLayout.inset)
            }
        }
    }
}

/// The window's top line, level with the traffic lights: the model's name (or its pop-up) and how it runs, then the
/// window's actions. It drags the window like a title bar.
struct ToolBand<Name: View, Trailing: View>: View {
    let detail: String
    @ViewBuilder let name: Name
    @ViewBuilder let trailing: Trailing
    @Environment(\.palette) private var p
    /// Kept for callers that read it off the type.
    static var leading: CGFloat { ToolLayout.bandLeading }

    init(detail: String, @ViewBuilder name: () -> Name, @ViewBuilder trailing: () -> Trailing) {
        self.detail = detail; self.name = name(); self.trailing = trailing()
    }

    var body: some View {
        HStack(spacing: 8) {
            name
            if !detail.isEmpty { Text(detail).font(.system(size: 12)).foregroundStyle(p.t3.swiftUI).lineLimit(1) }
            Spacer(minLength: 12)
            trailing
        }
        .padding(.leading, ToolLayout.bandLeading).padding(.trailing, 12)
        .frame(height: ToolWindowController.band)
        .contentShape(Rectangle())
        .gesture(WindowDragGesture())
    }
}

extension ToolBand where Name == ToolBandTitle {
    init(title: String, detail: String, @ViewBuilder trailing: () -> Trailing) {
        self.init(detail: detail, name: { ToolBandTitle(title: title) }, trailing: trailing)
    }
}

struct ToolBandTitle: View {
    let title: String
    var body: some View { Text(title).font(.system(size: 13, weight: .semibold)).lineLimit(1) }
}

/// The task's title, then the model it runs on in the secondary colour (a pop-up where the window offers a choice).
struct TaskTitle<Model: View>: View {
    let title: String
    @ViewBuilder let model: Model
    var body: some View {
        HStack(spacing: 8) {
            ToolBandTitle(title: title).layoutPriority(1).controlID("task-title")
            model
        }
    }
}

/// A one-model window's model name beside its task (Chat, Run): secondary, its precision in the hover.
struct ModelName: View {
    let model: ToolModel?
    @Environment(\.palette) private var p
    var body: some View {
        Text(model?.name ?? "").font(.system(size: 13)).foregroundStyle(p.t2.swiftUI).lineLimit(1)
            .contentShape(Rectangle()).windowHelp(model.map(ModelPicker.precisionLine)).controlID("model-name")
    }
}

/// Shows or hides the figures of every tool window (time, length, memory).
struct DetailsButton: View {
    @ToolDetailsShown(ToolDetails.key) private var details = false
    var body: some View {
        BandButton(symbol: "chart.bar.xaxis", help: details ? "Hide Run Details" : "Show Run Details", on: details) { details.toggle() }
            .controlID("details")
    }
}

/// A tool session's band: the model (a pop-up when the window offers a choice), its operations as pills, Show in Finder.
struct SessionBand: View {
    @ObservedObject var session: ToolSession
    /// The window picks among models (one window per tool); false: the window belongs to its one model.
    var picks = true
    var reveals = true
    /// A window's own band button before the pills (Transcribe's microphone).
    var accessory: AnyView?

    var body: some View {
        // The task names the window; the model is a choice inside it, its precision in the pop-up's hover.
        ToolBand(detail: "") {
            TaskTitle(title: session.taskTitle) {
                if picks {
                    ModelPicker(session: session).controlID("model")
                } else {
                    ModelName(model: session.model?.model)
                }
            }
        } trailing: {
            if let accessory { accessory }
            OperationPills(session: session)
            DetailsButton()
            // The window's one Show in Finder: the shown result's file, else the tool's folder.
            if reveals {
                BandButton(symbol: "folder", help: "Show in Finder (\(session.revealFolderName))") { session.revealInFinder() }.controlID("reveal")
            }
        }
    }
}

/// The band's model pop-up: the chosen model's name in the secondary colour, then every model the window offers (by kind
/// when there is more than one kind); a model that cannot run yet is listed with its reason and can be chosen (the
/// window then offers its Download or Approve…).
struct ModelPicker: View {
    @ObservedObject var session: ToolSession
    @Environment(\.palette) private var p

    var body: some View {
        Menu {
            let groups = Dictionary(grouping: session.models, by: { $0.model.kind })
            if groups.count > 1 {
                ForEach(groups.keys.sorted { Self.order($0) < Self.order($1) }, id: \.self) { kind in
                    Section(Self.heading(kind)) { ForEach(groups[kind] ?? []) { item($0) } }
                }
            } else {
                ForEach(session.models) { item($0) }
            }
        } label: {
            HStack(spacing: 4) {
                Text(session.model?.name ?? "Choose a model").font(.system(size: 13)).foregroundStyle(p.t2.swiftUI).lineLimit(1)
                Image(systemName: "chevron.down").font(.system(size: 9, weight: .semibold)).foregroundStyle(p.t3.swiftUI)
            }
            .padding(.vertical, 4).contentShape(Rectangle())
            // The pop-up says the model once, by its own label: the name drawn inside it is not a second element.
            .accessibilityHidden(true)
        }
        .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden).fixedSize()
        .disabled(session.running)
        .windowHelp(session.running ? "The model can change once this \(session.resultNoun) is done." : Self.help(session.model?.model))
        .accessibilityLabel("Model: \(session.model?.name ?? "none")")
    }

    private func item(_ m: ModelChoice) -> some View {
        Button {
            session.choose(m.id)
        } label: {
            if m.id == session.selected {
                Label(Self.title(m), systemImage: "checkmark")
            } else {
                Text(Self.title(m))
            }
        }.disabled(session.running)
    }

    /// The pop-up's hover: how the chosen model runs ("fp16 · Fast"), then what the pop-up does.
    static func help(_ model: ToolModel?) -> String {
        guard let model else { return "Choose the model" }
        return precisionLine(model) + "\nChoose the model"
    }
    /// "Kokoro 82M · fp16 · Fast".
    static func precisionLine(_ model: ToolModel) -> String { model.name + " \u{00b7} " + model.detail }

    static func title(_ m: ModelChoice) -> String {
        guard let why = m.unavailable else { return m.name }
        return m.name + " \u{2014} " + Self.short(why.message)
    }
    /// The reason's first sentence, for a menu line.
    static func short(_ message: String) -> String {
        let first = message.split(separator: ".", maxSplits: 1).first.map(String.init) ?? message
        return first.count > 60 ? String(first.prefix(59)) + "\u{2026}" : first
    }
    static func heading(_ kind: String) -> String {
        KindPresentation.title(kind) ?? (kind == "voices" ? "Voices" : kind == "sound" ? "Sound" : kind == "images" ? "Images" : OperationForm.title(kind))
    }
    static func order(_ kind: String) -> (Int, String) { (kind == "sound" ? 0 : kind == "voices" ? 1 : 2, kind) }
}

/// The chosen model's operations as pills, when it offers more than one.
struct OperationPills: View {
    @ObservedObject var session: ToolSession
    var body: some View {
        let ops = session.pillOperations
        if ops.count > 1 {
            HStack(spacing: 6) {
                ForEach(ops) { op in
                    Pill(title: session.pillTitle(op), on: op.name == session.operation, enabled: !session.running, help: op.title) { session.select(op.name) }
                        .controlID("op-\(op.name)")
                }
            }
        }
    }
    /// "Generate Music" → "Music": the pills sit beside each other, the verb is the run button's.
    static func title(_ title: String) -> String {
        title.hasPrefix("Generate ") && title.count > 9 ? String(title.dropFirst(9)) : title
    }
}

/// A borderless symbol button in the band (28 × 28, the browser's sidebar button).
struct BandButton: View {
    let symbol: String
    let help: String
    var enabled = true
    /// A mode that is on (live transcription, a JSON answer): drawn as a chosen pill is.
    var on = false
    let action: () -> Void
    @Environment(\.palette) private var p
    var body: some View {
        Button(action: action) {
            Image(systemName: symbol).font(.system(size: 14)).frame(width: 28, height: 28)
                .background((on ? p.accent.withAlphaComponent(0.2) : .clear).swiftUI, in: RoundedRectangle(cornerRadius: 6))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain).foregroundStyle((on ? p.accentText : enabled ? p.t2 : p.t4).swiftUI).disabled(!enabled)
        .windowHelp(help).accessibilityLabel(help)
    }
}

/// The browser's facet pill: 22 high, corner 6; on = accent tint.
struct Pill: View {
    let title: String
    var on = false
    var enabled = true
    let help: String
    let action: () -> Void
    @Environment(\.palette) private var p
    var body: some View {
        Button(action: action) {
            Text(title).font(.system(size: 12, weight: .medium).monospacedDigit())
                .foregroundStyle((enabled ? on ? p.accentText : p.t1 : p.dark ? p.t4 : p.t3).swiftUI)
                .padding(.horizontal, 10).frame(height: 24)
                // Disabled as the browser draws an unavailable modality: the field at half strength, the label in t4.
                .background(
                    (on ? p.accent.withAlphaComponent(0.2) : enabled ? p.field : p.field.withAlphaComponent(p.field.alphaComponent * 0.5)).swiftUI,
                    in: RoundedRectangle(cornerRadius: 6)
                )
                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(on ? p.accent.withAlphaComponent(0.55).swiftUI : .clear, lineWidth: 1))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain).disabled(!enabled).windowHelp(help).accessibilityLabel(help)
    }
}

/// A full-width button (32 high): the run in the accent, Stop neutral.
struct WideButton: View {
    let title: String
    var primary = true
    var enabled = true
    /// Why it is off, when it is.
    var help: String?
    /// Its hover while on (what "Anyway (slower)" means).
    var hover: String?
    let action: () -> Void
    @Environment(\.palette) private var p
    var body: some View {
        Button(action: action) {
            Text(title).font(.system(size: 13, weight: .semibold))
                .foregroundStyle(primary ? (enabled ? Color.white : p.t3.swiftUI) : p.t1.swiftUI)
                .frame(maxWidth: .infinity).frame(height: 32)
                .background((primary ? (enabled ? p.accent : p.selFill) : p.field).swiftUI, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(primary ? .clear : p.raisedBorder.swiftUI, lineWidth: 1))
                .contentShape(Rectangle())
        }.buttonStyle(.plain).disabled(!enabled).accessibilityLabel(title).windowHelp(enabled ? hover : nil)
            // A disabled button takes no hover: its reason is drawn over it on a layer that does.
            .accessibilityHint(enabled ? "" : help ?? "")
            .overlay { if !enabled, let help { Color.clear.contentShape(Rectangle()).windowHelp(help).accessibilityLabel(help) } }
    }
}

/// The run button, Stop while it runs.
struct RunButton: View {
    @ObservedObject var session: ToolSession
    let title: String
    /// Its control id ("generate", "remove", "run").
    let id: String
    /// Blue while running would make something new; quiet once it would make what is shown (`ToolSession.isFresh`).
    var body: some View {
        if session.running {
            WideButton(title: "Stop", primary: false, enabled: true) { session.stop() }.controlID("stop")
        } else if let prepare = session.preparesInstead, let action = prepare.action {
            // One way to fix it, in place: the button is the Download (or Approve…) the model needs first.
            WideButton(title: prepare.actionTitle ?? action.rawValue, primary: true, enabled: true) { session.performAction(action) }.controlID(id)
        } else {
            // After a crash or a memory refusal the button is the rerun ("Try Again", "Speak Anyway (slower)").
            WideButton(
                title: session.runButtonTitle(title), primary: !session.isFresh, enabled: session.canRun, help: session.notRunnable,
                hover: session.rerunInstead == .loadAnyway ? OperationProblem.anywayHelp : nil
            ) {
                session.runButtonAction()
            }.controlID(id)
        }
    }
}

/// A failure or a reason the tool cannot run, with the one action that fixes it.
struct ProblemLine: View {
    let problem: OperationProblem
    /// A failure is red; a state the model is in (not on this Mac, not approved) is amber.
    var failure = true
    /// False where another control already does the action.
    var showsAction = true
    let act: (OperationProblem.Action) -> Void
    @Environment(\.palette) private var p
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(problem.message, systemImage: failure ? "exclamationmark.triangle.fill" : "arrow.down.circle")
                .font(.system(size: 12)).foregroundStyle((failure ? p.red : p.amber).swiftUI)
                .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
                .windowHelp(problem.detail).controlID("problem")
            if showsAction, let action = problem.action {
                Button {
                    act(action)
                } label: {
                    Text(problem.actionTitle ?? action.rawValue).font(.system(size: 12, weight: .medium)).foregroundStyle(p.accentText.swiftUI)
                        .padding(.vertical, 4).contentShape(Rectangle())
                }.buttonStyle(.plain).windowHelp(action == .loadAnyway ? OperationProblem.anywayHelp : nil).controlID("action")
            }
        }
    }
}

/// The session's state line above the run button: why the model cannot run here, else the last failure not tied to a field.
struct SessionProblem: View {
    @Environment(\.palette) private var palette
    @ObservedObject var session: ToolSession
    var body: some View {
        if let blocked = session.blocked {
            // Its action is the run button's own when the button became it (Download, Approve…).
            ProblemLine(problem: blocked, failure: false, showsAction: session.preparesInstead == nil) { session.performAction($0) }
        } else if let problem = session.problem, problem.field == nil {
            // No Try Again beside a run button that tries again.
            ProblemLine(problem: session.worded(problem), showsAction: !session.duplicatesRun(problem)) { session.performAction($0) }
        } else if let notice = session.model?.model.memoryNotice {
            Text(notice).font(.system(size: 12)).foregroundStyle(palette.t2.swiftUI)
        }
    }
}
