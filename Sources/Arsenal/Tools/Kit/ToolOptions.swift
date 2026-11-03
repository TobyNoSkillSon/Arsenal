// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI
import UniformTypeIdentifiers

// The tool kit's options column, the same in every tool window: what the window works on (a lead, such as Remove
// Background's image), the main text, then the chosen operation's other options as its schema declares them (short
// ones in a grid, label left; longer ones as blocks, title above), the reason a run cannot start or the last failure,
// and the run button at the foot.

/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
private typealias OptionsHeight = SwiftUI.State<CGFloat>
private typealias VisibleHeight = SwiftUI.State<CGFloat>

struct OptionsColumn<Lead: View>: View {
    @ObservedObject var session: ToolSession
    let run: RunButton
    /// The lead takes the column when there is nothing else to set (Remove Background's image beside its cut-out).
    var leadFills = false
    /// Option groups drawn above the other options (a capability window's alternatives).
    var groups: AnyView?
    @ViewBuilder let lead: Lead
    /// The room under the last option when they scroll.
    static var foldInset: CGFloat { 16 }
    /// The options' own height: their scroll area is never taller than they are.
    @OptionsHeight private var optionsHeight: CGFloat = 0
    /// The height the scroll area got: below it the options continue, marked by a hairline and the scroll indicator.
    @VisibleHeight private var visibleHeight: CGFloat = 0

    var body: some View {
        // The clip to clone shows in Speak only after New Voice…; sampler knobs fold under Advanced.
        let shown = session.optionFields.filter(session.showsField)
        let fields = shown.filter { !session.isAdvanced($0) }
        let advanced = shown.filter(session.isAdvanced)
        // Cloning or designing a voice has no Voice field of its own: the picker that leads back to the voices stands in.
        let voiceRow = session.makingVoice && !(session.form?.fields.contains { $0.control == .voice } ?? false)
        let folds = optionsHeight > visibleHeight + 1 && visibleHeight > 0
        VStack(alignment: .leading, spacing: 14) {
            // Nothing cached to draw yet: a small spinner, so the column never looks broken.
            if session.loading, session.form == nil {
                ProgressView().controlSize(.small).frame(maxWidth: .infinity).padding(.top, 24).controlID("options-loading")
            }
            lead
            // The prompt gives up height (down to its minimum) before the options do.
            if session.showsMain, let main = session.form?.main { MainEditor(field: main, session: session) }
            if !shown.isEmpty || groups != nil || voiceRow {
                ScrollViewReader { proxy in
                    ScrollView(.vertical) {
                        VStack(alignment: .leading, spacing: 14) {
                            if let groups { groups }
                            if voiceRow { VoiceModeRow(session: session) }
                            OptionFields(fields: fields, session: session)
                            if !advanced.isEmpty {
                                AdvancedDisclosure(session: session)
                                if session.advancedOpen { OptionFields(fields: advanced, session: session) }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        // Room under the last control, so it scrolls fully clear of the edge above the run button.
                        .padding(.bottom, folds ? Self.foldInset : 0)
                        // Off while a run goes; the column still scrolls.
                        .disabled(session.running)
                        .onGeometryChange(for: CGFloat.self, of: { $0.size.height }, action: { optionsHeight = $0 })
                    }
                    .scrollIndicators(folds ? .visible : .automatic)
                    .frame(maxHeight: optionsHeight > 0 ? optionsHeight : nil)
                    .onGeometryChange(for: CGFloat.self, of: { $0.size.height }, action: { visibleHeight = $0 })
                    // More below: a hairline edge, never a fade over live controls.
                    .overlay(alignment: .bottom) { if folds { FoldEdge() } }
                    // A field's problem is scrolled into view, never left below the fold.
                    .onChange(of: session.problem) { _, problem in
                        guard let key = problem?.field else { return }
                        // A problem under a folded option opens Advanced first.
                        if advanced.contains(where: { $0.key == OptionFields.anchor(key) }) { session.advancedOpen = true }
                        // Only as far as needed: a field already in view leaves the controls above it in place.
                        DispatchQueue.main.async { proxy.scrollTo(OptionFields.anchor(key)) }
                    }
                    // A window opened onto a state that already has a field's problem shows it too.
                    .onAppear {
                        guard let key = session.problem?.field else { return }
                        DispatchQueue.main.async { proxy.scrollTo(OptionFields.anchor(key)) }
                    }
                }
                .layoutPriority(1)
            }
            if !(leadFills && shown.isEmpty) { Spacer(minLength: 0).layoutPriority(-1) }
            SessionProblem(session: session)
            run
        }
    }
}

/// The options' lower edge while more of them lies below: the window's hairline.
private struct FoldEdge: View {
    @Environment(\.palette) private var p
    var body: some View { Rectangle().fill(p.hairline.swiftUI).frame(height: 1).allowsHitTesting(false) }
}

/// The fold over the options a person rarely tunes (samplers; a voice's seed and steps): one row in the option labels'
/// font, its chevron turning as it opens. Closed in every new window.
struct AdvancedDisclosure: View {
    @ObservedObject var session: ToolSession
    @Environment(\.palette) private var p
    var body: some View {
        Button {
            session.advancedOpen.toggle()
        } label: {
            HStack(spacing: 5) {
                Text("Advanced").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI)
                Image(systemName: "chevron.right").font(.system(size: 9, weight: .semibold)).foregroundStyle(p.t3.swiftUI)
                    .rotationEffect(.degrees(session.advancedOpen ? 90 : 0))
                Spacer(minLength: 0)
            }
            .frame(height: 20).contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(session.advancedOpen ? "Hide advanced options" : "Show advanced options")
        .controlID("advanced")
    }
}

/// While a voice is cloned or designed: the Voice row, on the way it was chosen, leading back to the voices.
struct VoiceModeRow: View {
    @ObservedObject var session: ToolSession
    @Environment(\.palette) private var p
    var body: some View {
        HStack(spacing: 12) {
            Text("Voice").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI).lineLimit(1).fixedSize()
                .frame(minWidth: OptionFields.labelWidth, alignment: .leading)
            VoicePicker(session: session, fieldKey: nil, title: "Voice")
        }
    }
}

extension OptionsColumn where Lead == EmptyView {
    init(session: ToolSession, run: RunButton) { self.init(session: session, run: run, lead: { EmptyView() }) }
}

/// The main text: what to make or what to say.
struct MainEditor: View {
    let field: ToolField
    @ObservedObject var session: ToolSession
    @Environment(\.palette) private var p
    /// Its height when the options leave room, and the least it gives up to them.
    static let height: CGFloat = 150, minHeight: CGFloat = 96

    var body: some View {
        let text = Binding(get: { session.value(field.key).text }, set: { session.set(field.key, .text($0)) })
        let placeholder: String = {
            if case .main(let words) = field.control { return words }; return ""
        }()
        VStack(alignment: .leading, spacing: 6) {
            if session.mainSingleLine {
                // One line's worth (a search query): a field, not an editor.
                TextField(field.title, text: text, prompt: Text(placeholder).foregroundStyle(p.t3.swiftUI))
                    .textFieldStyle(.plain).font(.system(size: 13.5)).padding(.horizontal, 10).frame(height: 32)
                    .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
                    .disabled(session.running)
                    .controlID("field-\(field.key)")
            } else {
                editor(text: text, placeholder: placeholder)
            }
            FieldProblem(session: session, keys: [field.key])
        }
    }

    private func editor(text: Binding<String>, placeholder: String) -> some View {
        Group {
            ZStack(alignment: .topLeading) {
                TextEditor(text: text)
                    .font(.system(size: 13.5)).scrollContentBackground(.hidden).padding(.horizontal, 6).padding(.vertical, 8)
                if text.wrappedValue.isEmpty {
                    Text(placeholder).font(.system(size: 13.5)).foregroundStyle(p.t3.swiftUI).padding(.horizontal, 11).padding(.vertical, 8).allowsHitTesting(false)
                }
            }
            .frame(minHeight: Self.minHeight, idealHeight: Self.height, maxHeight: Self.height)
            .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
            .disabled(session.running)
            .controlID("field-\(field.key)")
        }
    }
}

/// The options under the main text: short ones in a grid (label left), the rest as blocks (title above).
struct OptionFields: View {
    let fields: [ToolField]
    @ObservedObject var session: ToolSession
    @Environment(\.palette) private var p
    static let labelWidth: CGFloat = 44

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // What the operation works on (its files) first, then the short settings, then the rest.
            // A song's Instrumental switch is drawn beside its lyrics' title.
            let lyrics = self.fields.contains { $0.control == .lyrics }
            let fields = lyrics ? self.fields.filter { $0.key != "instrumental" } : self.fields
            let content = fields.filter(Self.isContent)
            ForEach(content) { f in FieldControl(field: f, session: session).id(f.key) }
            let compact = fields.filter(\.compact)
            if !compact.isEmpty {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 10) {
                    ForEach(compact) { f in
                        GridRow {
                            Text(f.title).font(.system(size: 12)).foregroundStyle(p.t2.swiftUI).lineLimit(1).fixedSize()
                                .frame(minWidth: Self.labelWidth, alignment: .leading).contentShape(Rectangle()).windowHelp(f.help)
                            FieldControl(field: f, session: session).id(f.key)
                        }
                        if let problem = session.problem, problem.field == f.key {
                            GridRow {
                                Color.clear.frame(width: Self.labelWidth, height: 1)
                                FieldProblem.text(problem.message, p)
                            }
                        }
                    }
                }
            }
            ForEach(fields.filter { !$0.compact && !Self.isContent($0) }) { f in FieldControl(field: f, session: session).id(f.key) }
        }
    }
    /// The field a problem key belongs to ("reference.text" is the clip's transcript).
    static func anchor(_ key: String) -> String { key.hasSuffix(".text") ? String(key.dropLast(5)) : key }
    /// What the operation works on, drawn before the short settings: its files, a song's words, lists to rank or find.
    static func isContent(_ f: ToolField) -> Bool {
        switch f.control {
        case .generic(.file), .lyrics, .tags, .lines: return true
        default: return false
        }
    }
}

/// A field's own problem, under it.
struct FieldProblem: View {
    @ObservedObject var session: ToolSession
    let keys: [String]
    @Environment(\.palette) private var p
    var body: some View {
        if let problem = session.problem, let field = problem.field, keys.contains(field) { Self.text(problem.message, p) }
    }
    static func text(_ message: String, _ p: ArsenalPalette) -> some View {
        Text(message).font(.system(size: 11.5)).foregroundStyle(p.red.swiftUI).fixedSize(horizontal: false, vertical: true)
    }
}

/// One option as its schema calls for it.
struct FieldControl: View {
    let field: ToolField
    @ObservedObject var session: ToolSession
    /// False inside a chooser, whose pill names it.
    var titled = true
    @Environment(\.palette) private var p
    @Environment(\.isEnabled) private var enabled

    private var text: Binding<String> { Binding(get: { session.value(field.key).text }, set: { session.set(field.key, .text($0)) }) }
    private var instrumental: Bool {
        if case .toggle(true) = session.value("instrumental") { return true }; return false
    }
    private var hasInstrumental: Bool { session.form?.fields.contains { $0.key == "instrumental" } ?? false }
    /// The language of a song with lyrics (not a transcription's or a translation's).
    private var sungLanguage: Bool { field.key == "language" && (session.form?.fields.contains { $0.control == .lyrics } ?? false) }
    /// The figure beside a stepper (Length, Steps): one width, so the steppers line up.
    static let figureWidth: CGFloat = 44

    var body: some View {
        switch field.control {
        case .main: EmptyView()
        case .voice: voice
        case .seed: seed
        case .seconds(let lo, let hi): seconds(lo, hi)
        case .size(let sizes): size(sizes)
        case .note: block { editor(height: 48, placeholder: field.key == "negative_prompt" ? "What to leave out" : "") }
        case .reference(let transcript): block(extra: [ToolForm.transcriptKey(field.key)]) { clip(transcript) }
        case .images(let most): block { well(FileAccept(mimes: ["image/*"]), multiple: true, most: most) }
        case .language(let choices):
            // A song's language goes with its words: off while the song is instrumental.
            LanguagePicker(field: field, choices: choices, session: session).disabled(sungLanguage && instrumental)
        case .lyrics:
            // The instrumental switch sits beside the lyrics' title; on, the words are off (an instrumental has none).
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Text(field.title).font(.system(size: 12)).foregroundStyle(p.t2.swiftUI).lineLimit(1).windowHelp(Self.help(field))
                    Spacer(minLength: 4)
                    if hasInstrumental {
                        Toggle(isOn: Binding(get: { instrumental }, set: { session.set("instrumental", .toggle($0)) })) {
                            Text("Instrumental").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI)
                        }.toggleStyle(.switch).controlSize(.mini).controlID("field-instrumental")
                    }
                }
                editor(height: 110, placeholder: instrumental ? "Instrumental" : "Words to sing").disabled(instrumental).opacity(instrumental ? 0.5 : 1)
                FieldProblem(session: session, keys: [field.key])
            }
        case .tags: block { TagField(field: field, session: session, placeholder: Self.tagPlaceholders[field.key] ?? "Type and press Return") }
        case .lines: block { LinesEditor(field: field, session: session) }
        case .pages: line(placeholder: "All")
        case .numbers: block { line(placeholder: "0.25, 0.5, 0.75", mono: true) }
        case .schema: block { SchemaEditor(key: field.key, text: text, startsEmpty: field.startsEmpty) }
        case .points, .boxes: block { editor(height: 72, placeholder: "JSON", mono: true) }
        case .generic(let control): generic(control)
        }
    }

    /// A field with its title above it and its own problem under it.
    private func block<C: View>(extra: [String] = [], @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if titled { Text(field.title).font(.system(size: 12)).foregroundStyle(p.t2.swiftUI).lineLimit(1).windowHelp(Self.help(field)) }
            content()
            FieldProblem(session: session, keys: [field.key] + extra)
        }
    }

    // MARK: Named controls

    private var voice: some View { VoicePicker(session: session, fieldKey: field.key, title: field.title, help: field.help) }
    /// The tag of a pop-up line that only explains, never chosen.
    static let unlisted = VoicePicker.unlisted

    /// Random when empty; Reuse last takes the shown result's seed.
    private var seed: some View {
        HStack(spacing: 6) {
            line(placeholder: "Random", mono: true)
            Button {
                session.set(field.key, .text(session.shownSeed.map(String.init) ?? ""))
            } label: {
                Text("Reuse last").font(.system(size: 12, weight: .medium)).padding(.horizontal, 8).frame(height: 26).contentShape(Rectangle())
            }.buttonStyle(.plain).foregroundStyle((session.shownSeed == nil ? p.t4 : p.accentText).swiftUI).disabled(session.shownSeed == nil)
                .windowHelp("Use the shown \(session.resultNoun)\u{2019}s seed: the same inputs then make the same \(session.resultNoun).").controlID("seed-back")
        }
    }

    /// Drawn as Steps is ("12 s" and a stepper), so the column reads as one form; the figure can also be typed.
    private func seconds(_ lo: Double, _ hi: Double) -> some View {
        let range = max(1, Int(lo.rounded(.up)))...max(1, Int(lo.rounded(.up)), Int(hi.rounded(.down)))
        let whole = Binding(
            get: { min(max(Int((Double(session.value(field.key).text) ?? field.defaultValue?.number ?? 1).rounded()), range.lowerBound), range.upperBound) },
            set: { session.set(field.key, .text(String($0))) })
        let placeholder = field.defaultValue?.number.map(OperationForm.number) ?? ""
        let shown = text.wrappedValue.isEmpty ? placeholder : text.wrappedValue
        return HStack(spacing: 8) {
            HStack(spacing: 3) {
                TextField(field.title, text: text, prompt: Text(placeholder).foregroundStyle(p.t3.swiftUI)).textFieldStyle(.plain).font(.system(size: 13).monospacedDigit())
                    .frame(width: max(16, CGFloat(shown.count) * 8 + 1))
                    .controlID("field-\(field.key)")
                Text("s").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI)
            }.frame(width: Self.figureWidth, alignment: .leading)
            Stepper(field.title, value: whole, in: range).labelsHidden().controlID("step-\(field.key)")
            Spacer(minLength: 0)
        }
        .windowHelp("Up to \(OperationForm.number(hi)) s; type a length or step it")
    }

    /// The sizes the operation offers; a free size from the model's settings is listed too.
    private func size(_ sizes: ImageSizes) -> some View {
        let current = session.value(field.key).text
        let extra = ImageSize(resolution: current).flatMap { sizes.sizes.contains($0) ? nil : $0 }
        return Picker(field.title, selection: Binding(get: { current }, set: { session.set(field.key, .text($0)) })) {
            if sizes.reference { Text(ImageSizes.sameAsReference).tag("") }
            ForEach(sizes.sizes) { s in Text(s.title).tag(s.resolution) }
            if let extra { Text(extra.title).tag(extra.resolution) }
        }.labelsHidden().pickerStyle(.menu).frame(maxWidth: .infinity, alignment: .leading).controlID("field-\(field.key)")
    }

    static let transcriptPlaceholder = "Type what the clip says"
    /// A voice to clone: its clip, then what the clip says once there is one.
    @ViewBuilder private func clip(_ transcript: Bool?) -> some View {
        well(FileAccept(mimes: ["audio/*"]), multiple: false, most: 1, help: "A short clip of the voice to speak like: WAV, FLAC, MP3 or M4A")
        if let transcript, !session.value(field.key).files.isEmpty {
            let key = ToolForm.transcriptKey(field.key)
            // One placeholder, no label above it: a missing transcript's line under the field says why it is needed.
            TextField(
                "What the clip says",
                text: Binding(get: { session.value(key).text }, set: { session.set(key, .text($0)) }),
                prompt: Text(Self.transcriptPlaceholder).foregroundStyle(p.t3.swiftUI)
            )
            .windowHelp(transcript ? "This model needs the clip\u{2019}s words to copy the voice." : "The clip\u{2019}s words, if you know them: the copy is closer.")
            .textFieldStyle(.plain).font(.system(size: 13))
            .padding(.horizontal, 8).frame(height: 26)
            .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
            .controlID("field-\(key)")
        }
    }

    // MARK: Generic

    @ViewBuilder private func generic(_ control: FormField.Control) -> some View {
        switch control {
        case .text(let multiline):
            if multiline {
                block { editor(height: 72, placeholder: "") }
            } else {
                line(placeholder: field.defaultValue?.string ?? Self.placeholders[field.key] ?? (field.key == "language" ? "Automatic" : ""))
            }
        case .integer(let lo, let hi, let stepper):
            if stepper, let lo, let hi {
                // Unset, nothing is sent: the figure is the declared default, else "Auto", dim either way; the range is
                // the hover.
                let unset = Int(session.value(field.key).text) == nil
                let declared = field.defaultValue?.number?.roundedInt
                let rest = declared ?? lo
                HStack(spacing: 8) {
                    Text(unset ? declared.map { String($0) + (field.unit ?? "") } ?? "Auto" : session.value(field.key).text + (field.unit ?? ""))
                        .font(.system(size: 13).monospacedDigit())
                        .foregroundStyle((enabled && !unset ? p.t1 : p.t3).swiftUI)
                        .frame(width: Self.figureWidth, alignment: .leading)
                        // The whole figure cell carries its hover, not only the digits' glyphs.
                        .contentShape(Rectangle())
                        .windowHelp(Self.rangeHelp(unset: unset, declared: declared != nil, Self.range(lo, hi)))
                    Stepper(
                        field.title,
                        value: Binding(
                            get: { min(max(Int(session.value(field.key).text) ?? rest, lo), hi) },
                            set: { session.set(field.key, .text(String($0))) }), in: lo...hi
                    ).labelsHidden().controlID("field-\(field.key)")
                    Spacer(minLength: 0)
                }
            } else {
                HStack(spacing: 6) {
                    line(
                        placeholder: field.defaultValue?.number.map(OperationForm.number) ?? Self.placeholders[field.key] ?? "Auto", mono: true,
                        help: Self.rangeHelp(unset: true, declared: field.defaultValue != nil, Self.range(lo, hi)))
                    if let unit = field.unit { Text(unit).font(.system(size: 12)).foregroundStyle(p.t2.swiftUI).fixedSize() }
                }
            }
        case .number(let lo, let hi):
            // A slider for a bounded range a drag can set (a threshold, guidance); a huge range is typed.
            if let lo, let hi, lo < hi, hi - lo <= 1000 {
                // Unset, nothing is sent: the knob rests dimmed at the model's declared default and the figure says so in
                // grey ("Auto (0.9)"), so the knob and the figure agree; with no declared default there is no knob and the
                // figure is "Auto". The default and the range are the figure's hover.
                let unset = Double(session.value(field.key).text) == nil
                let declared = field.defaultValue?.number
                let rest = Self.restingValue(key: field.key, declared: declared, lo: lo, hi: hi)
                let range = OperationForm.number(lo) + "\u{2013}" + OperationForm.number(hi)
                HStack(spacing: 8) {
                    ToolSlider(
                        value: Binding(
                            get: { min(max(Double(session.value(field.key).text) ?? rest, lo), hi) },
                            set: { session.set(field.key, .text(OperationForm.number(($0 * 100).rounded() / 100))) }), range: lo...hi, title: field.title,
                        neutral: unset, showsKnob: !unset || declared != nil
                    ).controlID("field-\(field.key)")
                    Text(unset ? Self.autoFigure(declared: declared, lo: lo, hi: hi) : session.value(field.key).text).font(.system(size: 12.5).monospacedDigit())
                        .foregroundStyle((enabled && !unset ? p.t1 : p.t3).swiftUI)
                        .frame(minWidth: 36, alignment: .trailing).fixedSize().contentShape(Rectangle())
                        // On the figure only: a tooltip cell over the slider would take its clicks (TooltipCell).
                        .windowHelp(Self.rangeHelp(unset: unset, declared: declared != nil, range, defaultValue: declared.map(OperationForm.number)))
                }
            } else {
                line(
                    placeholder: field.defaultValue?.number.map(OperationForm.number) ?? Self.placeholders[field.key] ?? "Auto", mono: true,
                    help: Self.rangeHelp(
                        unset: true, declared: field.defaultValue != nil, lo.flatMap { l in hi.map { OperationForm.number(l) + "\u{2013}" + OperationForm.number($0) } } ?? ""))
            }
        case .choice(let choices):
            // Unset sends nothing: "Auto" when that leaves it to the model; a declared default is listed by its value.
            let auto = !field.required && (field.defaultValue.map { !choices.contains($0) } ?? true)
            Picker(
                field.title,
                selection: Binding(
                    get: { Self.choiceIndex(session.value(field.key), choices: choices, auto: auto, defaultValue: field.defaultValue) },
                    set: { session.set(field.key, .choice($0 < 0 ? nil : $0)) })
            ) {
                if auto { Text("Auto").tag(-1) }
                ForEach(Array(choices.enumerated()), id: \.offset) { i, c in
                    Text(
                        field.choiceTitles[ToolFiles.scalar(c)] ?? c.string.map(Self.choiceTitle)
                            ?? (field.key == "scale" ? ToolFiles.scalar(c) + "\u{00d7}" : ToolFiles.scalar(c))
                    ).tag(i)
                }
            }.labelsHidden().pickerStyle(.menu).frame(maxWidth: .infinity, alignment: .leading).controlID("field-\(field.key)")
        case .toggle:
            VStack(alignment: .leading, spacing: 6) {
                Toggle(
                    isOn: Binding(
                        get: {
                            if case .toggle(let on) = session.value(field.key) { return on }; return false
                        }, set: { session.set(field.key, .toggle($0)) })
                ) {
                    Text(field.title).font(.system(size: 12)).foregroundStyle(p.t2.swiftUI)
                }.toggleStyle(.switch).controlSize(.small)
                    // Dimmed with the pickers while a run goes: the switch itself draws the same on or off.
                    .opacity(enabled ? 1 : 0.5)
                    .windowHelp(Self.help(field)).controlID("field-\(field.key)")
                FieldProblem(session: session, keys: [field.key])
            }
        case .file(let accept, let multiple, let most, _):
            block { well(accept, multiple: multiple, most: most) }
        case .json:
            block { editor(height: 72, placeholder: "JSON", mono: true) }
        }
    }

    /// A choice picker's selection: the chosen index; unset (or an index the choices no longer have), "Auto" (-1) when
    /// the picker lists it, else the declared default, else the first choice. A picker never shows a blank.
    static func choiceIndex(_ value: FieldValue, choices: [JSONValue], auto: Bool, defaultValue: JSONValue?) -> Int {
        if case .choice(let i?) = value, choices.indices.contains(i) { return i }
        if auto { return -1 }
        return defaultValue.flatMap { choices.firstIndex(of: $0) } ?? (choices.isEmpty ? -1 : 0)
    }

    /// "256–1536": a bounded whole number's range, for its hover.
    static func range(_ lo: Int?, _ hi: Int?) -> String {
        guard let lo, let hi else { return "" }
        return "\(lo)\u{2013}\(hi)"
    }
    /// What an unset field means and what it takes: "Auto: the model decides. Takes 1–2048.", "Auto: the model's
    /// default, 0.9. Takes 0–2."
    static func rangeHelp(unset: Bool, declared: Bool, _ range: String, defaultValue: String? = nil) -> String? {
        let takes = range.isEmpty ? "" : "Takes \(range)."
        guard unset else { return takes.isEmpty ? nil : takes }
        let auto = declared ? (defaultValue.map { "Auto: the model\u{2019}s default, \($0)." } ?? "The model\u{2019}s default.") : "Auto: the model decides."
        return (auto + " " + takes).trimmingCharacters(in: .whitespaces)
    }
    /// An unset slider's figure: "Auto (0.9)" at the declared default (where the knob rests), else "Auto" (no knob).
    static func autoFigure(declared: Double?, lo: Double, hi: Double) -> String {
        guard let declared else { return "Auto" }
        return "Auto (\(OperationForm.number(min(max(declared, lo), hi))))"
    }
    /// Where an unset slider's knob rests: the declared default, else the sampler's neutral value (no change to the
    /// model's choices), else the low end.
    static func restingValue(key: String, declared: Double?, lo: Double, hi: Double) -> Double {
        let neutral: [String: Double] = ["temperature": 1, "top_p": 1, "repetition_penalty": 1, "min_p": 0, "typical_p": 1, "presence_penalty": 0, "frequency_penalty": 0]
        let v = declared ?? neutral[key] ?? lo
        return min(max(v, lo), hi)
    }
    /// Sampler knobs in plain words (their schema text is jargon); any other field keeps its schema's help.
    static let plainHelp: [String: String] = [
        "top_k": "How many of the likeliest next choices it picks from. Fewer is safer, more is more varied.",
        "top_p": "Picks only from the likeliest choices that together make up this share. Lower is safer, higher is more varied.",
        "min_p": "Leaves out choices much less likely than the best one. Higher is safer.",
        "temperature": "How adventurous it is. Lower is steadier and more predictable, higher is more varied.",
        "repetition_penalty": "How much it avoids repeating itself. Above 1 repeats less.",
        "presence_penalty": "Nudges it toward new topics rather than ones already used.",
        "frequency_penalty": "Nudges it away from words it has already used often."
    ]
    static func help(_ field: ToolField) -> String? { plainHelp[field.key] ?? field.help }
    /// Words for an empty field whose emptiness has a meaning.
    static let tagPlaceholders = ["stems": "vocals, drums, bass", "labels": "Object names, then Return"]
    static let placeholders: [String: String] = [
        "focal_length": "Estimated", "start_frame": "First", "end_frame": "Last", "pages": "All", "seed": "Random", "top_n": "All",
        // Find Objects' modes say what each one takes.
        "phrase": "A description to find", "instruction": "A screen element to find"
    ]

    /// A language code reads as its name ("en" → English); a plain lowercase word starts with a capital, as a menu line
    /// does ("english" → English, "fine" → Fine); any other choice as it is ("birefnet-general", "512x512").
    /// Words joined by underscores read as words ("distorted_layout" → Distorted layout); a few values a person would not
    /// read as words have their own titles ("ocr" → Full page, a document reader's whole-page mode).
    static func choiceTitle(_ s: String) -> String {
        if let own = ownChoiceTitles[s] { return own }
        // A code macOS does not name stays a word (Locale gives "ocr" back as it is).
        let language = VoiceChoice.languageName(s)
        if s.count <= 6, s.range(of: "^[a-z]{2,3}(-[A-Za-z]{2})?$", options: .regularExpression) != nil, !language.isEmpty, language != s {
            return language
        }
        guard s.range(of: "^[a-z]+(_[a-z]+)*$", options: .regularExpression) != nil else { return s }
        let words = s.replacingOccurrences(of: "_", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }
    static let ownChoiceTitles = ["translate": "Translate to English", "html": "HTML", "ocr": "Full page"]

    private func well(_ accept: FileAccept, multiple: Bool, most: Int, help: String = FileWell.help) -> some View {
        FileWell(
            key: field.key, title: field.title, files: session.value(field.key).files, accept: accept, multiple: multiple, most: most, help: help,
            targeted: Binding(
                get: { session.dropTarget == field.key }, set: { session.dropTarget = $0 ? field.key : (session.dropTarget == field.key ? nil : session.dropTarget) }),
            player: session.player, enabled: !session.running,
            add: { session.addFiles($0, to: field.key) }, remove: { session.removeFile($0, from: field.key) })
    }

    /// Empty, the placeholder is dim (never mistaken for a typed value).
    private func line(placeholder: String, mono: Bool = false, help: String? = nil) -> some View {
        TextField(field.title, text: text, prompt: Text(placeholder).foregroundStyle(p.t3.swiftUI))
            .textFieldStyle(.plain).font(mono ? .system(size: 13).monospacedDigit() : .system(size: 13))
            .windowHelp(help)
            .padding(.horizontal, 8).frame(height: 26)
            .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
            .controlID("field-\(field.key)")
    }

    private func editor(height: CGFloat, placeholder: String, mono: Bool = false) -> some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: text)
                .font(mono ? .system(size: 11.5, design: .monospaced) : .system(size: 13)).scrollContentBackground(.hidden)
                .padding(.horizontal, 6).padding(.vertical, 6)
            if text.wrappedValue.isEmpty, !placeholder.isEmpty {
                Text(placeholder).font(.system(size: 13)).foregroundStyle(p.t3.swiftUI).padding(.horizontal, 11).padding(.vertical, 6).allowsHitTesting(false)
            }
        }
        .frame(height: height)
        .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
        .controlID("field-\(field.key)")
    }
}

// MARK: Slider

/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
private typealias SliderWidth = SwiftUI.State<CGFloat>

/// A slider whose knob reads on any ground: a white knob with an edge and a shadow over an accent fill, the same in
/// light and dark and in a window that is not key (the system knob vanished on the light field ground). Arrow keys and
/// VoiceOver step it by a twentieth of its range.
struct ToolSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    let title: String
    /// Unset ("Auto"): no fill and a dimmed knob at the default, until it is moved.
    var neutral = false
    /// False while unset with no declared default: no knob, since no position would be true ("Auto" alone).
    var showsKnob = true
    static let restingOpacity = 0.45
    @Environment(\.palette) private var p
    @Environment(\.isEnabled) private var enabled
    static let knob: CGFloat = 15

    @SliderWidth private var width: CGFloat = 0
    private var span: CGFloat { max(1, width - Self.knob) }

    var body: some View {
        let x = span * fraction
        ZStack(alignment: .leading) {
            Capsule().fill(p.t4.swiftUI).frame(height: 4).padding(.horizontal, Self.knob / 2)
            Capsule().fill(p.accent.swiftUI).frame(width: neutral ? 0 : x, height: 4).padding(.leading, Self.knob / 2)
            // Unset ("Auto"): the knob rests dimmed at the model's default with no fill, so it reads as not set yet, never
            // as a dead track; a click places it.
            Circle().fill(Color.white)
                .overlay(Circle().strokeBorder(Color.black.opacity(0.22), lineWidth: 0.5))
                .shadow(color: .black.opacity(neutral ? 0 : 0.3), radius: 1.5, y: 0.5)
                .frame(width: Self.knob, height: Self.knob).offset(x: x)
                .opacity(showsKnob ? (neutral ? Self.restingOpacity : 1) : 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading).frame(height: 20)
        .onGeometryChange(for: CGFloat.self, of: { $0.size.width }, action: { width = $0 })
        // A click sets the value where it lands and a drag follows the pointer, through an AppKit view: it takes the
        // first click of an inactive window, as the system slider does, and the window's tooltip cells cannot take it.
        .overlay(SliderTracker(enabled: enabled) { set(at: $0) })
        .opacity(enabled ? 1 : 0.5)
        // To accessibility it is the system slider (AXSlider, labelled, stepped by a twentieth of its range).
        .accessibilityRepresentation {
            Slider(value: Binding(get: { value }, set: { set($0) }), in: range, step: max((range.upperBound - range.lowerBound) / 20, .ulpOfOne)) { Text(title) }
                .accessibilityLabel(title)
                .accessibilityValue(OperationForm.number((value * 100).rounded() / 100))
        }
    }

    private var fraction: CGFloat {
        let width = range.upperBound - range.lowerBound
        return width > 0 ? CGFloat(min(max((value - range.lowerBound) / width, 0), 1)) : 0
    }
    private func set(at x: CGFloat) {
        set(range.lowerBound + Double((x - Self.knob / 2) / span) * (range.upperBound - range.lowerBound))
    }
    private func set(_ v: Double) { value = min(max(v, range.lowerBound), range.upperBound) }
}

/// The slider's pointer: mouse down and drags, as x in its own frame.
private struct SliderTracker: NSViewRepresentable {
    /// Off while a run is made: SwiftUI's disabled state does not reach an AppKit view by itself.
    let enabled: Bool
    let moved: (CGFloat) -> Void

    final class Tracker: NSView {
        var enabled = true
        var moved: (CGFloat) -> Void = { _ in }
        override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
        override func hitTest(_ point: NSPoint) -> NSView? { enabled ? super.hitTest(point) : nil }
        override func mouseDown(with event: NSEvent) { track(event) }
        override func mouseDragged(with event: NSEvent) { track(event) }
        private func track(_ event: NSEvent) { moved(convert(event.locationInWindow, from: nil).x) }
    }
    func makeNSView(context: Context) -> Tracker { Tracker() }
    func updateNSView(_ view: Tracker, context: Context) { view.enabled = enabled; view.moved = moved }
}

// MARK: Files

/// A file input: a drop well that is also the button to choose files. One file: the chosen image over a checkerboard
/// (click to choose another, × to clear), a clip with Play, or a document by name. Several: their thumbnails in a row,
/// each with ×, and a tile to add more up to the limit.
struct FileWell: View {
    let key: String
    let title: String
    let files: [URL]
    let accept: FileAccept
    var multiple = false
    var most = 1
    var help = FileWell.help
    var chosenHelp: String?
    @Binding var targeted: Bool
    @ObservedObject var player: SoundPlayer
    var enabled = true
    /// A single image's well height; nil: it takes the room it is given.
    var height: CGFloat? = 150
    /// The single image's upright size when the caller knows it; nil: read from the cached image.
    var imageSize: PixelSize?
    let add: ([URL]) -> Void
    let remove: (URL) -> Void
    /// Choose with a panel (tests answer it); nil: the open panel.
    var choose: (() -> [URL])?
    /// Drops other than file URLs (image bytes); nil: file URLs only.
    var receive: (([NSItemProvider]) -> Bool)?
    @Environment(\.palette) private var p

    static let help = "Drop a file here, or click to choose"
    private var audio: Bool { !accept.mimes.isEmpty && accept.mimes.allSatisfy { $0.hasPrefix("audio/") } }
    private var images: Bool { accept.images || accept.mimes.isEmpty }

    var body: some View {
        Group {
            if multiple {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        ForEach(files, id: \.self) { tile($0) }
                        if files.count < most { addTile }
                    }
                }
            } else if audio {
                if let clip = files.first { clipRow(clip) } else { clipWell }
            } else {
                single.frame(height: height).frame(maxHeight: height == nil ? .infinity : nil)
            }
        }
        .onDrop(of: receive == nil ? [.fileURL] : [.fileURL, .image], isTargeted: $targeted) { providers in
            guard enabled else { return false }
            if let receive { return receive(providers) }
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in add([url]) }
                }
            }
            return true
        }
    }

    private func pick() {
        if let choose { let urls = choose(); if !urls.isEmpty { add(urls) }; return }
        let panel = NSOpenPanel()
        let types = accept.mimes.compactMap { UTType(mimeType: $0) ?? ($0.hasPrefix("image/") ? .image : $0.hasPrefix("audio/") ? .audio : $0.hasPrefix("video/") ? .movie : nil) }
        if !types.isEmpty { panel.allowedContentTypes = types }
        panel.allowsMultipleSelection = multiple
        if panel.runModal() == .OK { add(panel.urls) }
    }

    private var border: Color { (targeted ? p.accent : p.raisedBorder).swiftUI }
    private func dashed(_ radius: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: radius).strokeBorder(border, style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
    }
    private var wash: Color { (targeted ? p.accent.withAlphaComponent(0.12) : .clear).swiftUI }
    private var ink: Color { (targeted ? p.accentText : p.t3).swiftUI }

    // MARK: One file

    private var single: some View {
        ZStack(alignment: .topTrailing) {
            Button(action: pick) {
                Group {
                    if let file = files.first { chosen(file) } else { empty }
                }.contentShape(Rectangle())
            }
            .buttonStyle(.plain).disabled(!enabled)
            .windowHelp(files.isEmpty ? help : chosenHelp ?? files.first?.lastPathComponent)
            .accessibilityLabel(files.isEmpty ? "Choose \(title.lowercased())" : "Choose another \(title.lowercased())")
            .controlID("well-\(key)")
            if let file = files.first, enabled {
                Button {
                    remove(file)
                } label: {
                    Image(systemName: "xmark.circle.fill").font(.system(size: 14)).symbolRenderingMode(.palette)
                        .foregroundStyle(Color.white, Color.black.opacity(0.6)).frame(width: 26, height: 26).contentShape(Rectangle())
                }.buttonStyle(.plain).windowHelp("Remove").accessibilityLabel("Remove \(file.lastPathComponent)").controlID("remove-\(key)-\(file.lastPathComponent)")
            }
        }
    }

    /// The chosen image at the largest size that fits, the checkerboard only under the image itself: an opaque photo
    /// shows none, a PNG with transparency shows its own. Another file by its name.
    @ViewBuilder private func chosen(_ url: URL) -> some View {
        Group {
            if images || ToolFiles.isImageFile(url) {
                FittedImage(url: url, size: imageSize, label: url.lastPathComponent).padding(8)
            } else {
                VStack(spacing: 4) {
                    Image(systemName: "doc").font(.system(size: 18)).foregroundStyle(p.t2.swiftUI)
                    Text(url.lastPathComponent).font(.system(size: 11.5)).lineLimit(1).truncationMode(.middle).foregroundStyle(p.t2.swiftUI)
                }.padding(6)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(p.sidebar.swiftUI)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(border, lineWidth: targeted ? 2 : 1))
    }

    private var empty: some View {
        VStack(spacing: 8) {
            Image(systemName: images ? "photo.badge.plus" : "doc.badge.plus").font(.system(size: 26, weight: .light))
            Text(images ? "Drop an image, or click to choose" : "Drop a file, or click to choose").font(.system(size: 12))
        }
        .foregroundStyle(ink)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(wash, in: RoundedRectangle(cornerRadius: 8))
        .overlay(dashed(8))
    }

    // MARK: A clip

    private func clipRow(_ url: URL) -> some View {
        HStack(spacing: 8) {
            Button {
                player.toggle(url)
            } label: {
                Image(systemName: player.playing == url ? "stop.fill" : "play.fill").font(.system(size: 11)).frame(width: 24, height: 24).contentShape(Rectangle())
            }.buttonStyle(.plain).foregroundStyle(p.t1.swiftUI).accessibilityLabel(player.playing == url ? "Stop the clip" : "Play the clip")
                .controlID("play-\(key)")
            Text(url.lastPathComponent).font(.system(size: 12.5)).lineLimit(1).truncationMode(.middle).windowHelp(url.path)
            Spacer(minLength: 4)
            Button {
                remove(url)
            } label: {
                Image(systemName: "xmark.circle.fill").font(.system(size: 13)).foregroundStyle(p.t3.swiftUI).frame(width: 24, height: 24).contentShape(Rectangle())
            }.buttonStyle(.plain).windowHelp("Remove").accessibilityLabel("Remove \(url.lastPathComponent)").controlID("remove-\(key)")
        }
        .padding(.horizontal, 6).frame(height: 40)
        .background(p.raised.swiftUI, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(border, lineWidth: 1))
    }

    private var clipWell: some View {
        Button(action: pick) {
            HStack(spacing: 8) {
                Image(systemName: "waveform.badge.plus").font(.system(size: 15))
                Text("Drop a clip, or click to choose").font(.system(size: 12))
            }
            .foregroundStyle(ink)
            .frame(maxWidth: .infinity).frame(height: 40)
            .background(wash, in: RoundedRectangle(cornerRadius: 8))
            .overlay(dashed(8))
            .contentShape(Rectangle())
        }.buttonStyle(.plain).disabled(!enabled).windowHelp(help).accessibilityLabel("Choose \(title.lowercased())").controlID("well-\(key)")
    }

    // MARK: Several files

    private func tile(_ url: URL) -> some View {
        ZStack(alignment: .topTrailing) {
            Group {
                if ToolFiles.isImageFile(url) {
                    Thumbnail(url: url)
                } else {
                    Image(systemName: "doc").font(.system(size: 18)).foregroundStyle(p.t2.swiftUI).frame(maxWidth: .infinity, maxHeight: .infinity).background(p.sidebar.swiftUI)
                }
            }
            .frame(width: 56, height: 56).clipShape(RoundedRectangle(cornerRadius: 6))
            .accessibilityLabel(url.lastPathComponent)
            if enabled {
                Button {
                    remove(url)
                } label: {
                    Image(systemName: "xmark.circle.fill").font(.system(size: 13)).symbolRenderingMode(.palette)
                        .foregroundStyle(Color.white, Color.black.opacity(0.6)).frame(width: 20, height: 20).contentShape(Rectangle())
                }.buttonStyle(.plain).windowHelp("Remove").accessibilityLabel("Remove \(url.lastPathComponent)").controlID("remove-\(key)-\(url.lastPathComponent)")
            }
        }
    }

    private var addTile: some View {
        Button(action: pick) {
            VStack(spacing: 3) {
                Image(systemName: images ? "photo.badge.plus" : "doc.badge.plus").font(.system(size: 16))
                if files.isEmpty { Text(images ? "Drop images, or click to choose" : "Drop files, or click to choose").font(.system(size: 11)) }
            }
            .foregroundStyle(ink)
            .frame(maxWidth: files.isEmpty ? .infinity : 56).frame(height: 56)
            .background(wash, in: RoundedRectangle(cornerRadius: 6))
            .overlay(dashed(6))
            .contentShape(Rectangle())
        }.buttonStyle(.plain).disabled(!enabled).windowHelp(help).accessibilityLabel("Add \(title.lowercased())").controlID("well-\(key)")
    }
}

/// The Voice pop-up: Default, the model's voices (once it is loaded) in labelled groups, then the ways to a new voice
/// (New Voice…, or Clone a Voice… and Design a Voice…). Before the model is loaded its voices are unknown: the chosen
/// voice (or Default) stays listed by name, with a line saying when the rest appear. Without a field (`fieldKey` nil)
/// it stands for a voice being made: New Voice… is its value, and a voice chosen goes back to Speak with it.
struct VoicePicker: View {
    @ObservedObject var session: ToolSession
    let fieldKey: String?
    let title: String
    var help: String?

    static let unlisted = "\u{0}unlisted"
    static let newVoicePrefix = "\u{0}new:"
    static let fromClip = "\u{0}clip"

    var body: some View {
        let current = fieldKey.map { session.value($0).text } ?? ""
        let name = session.model?.name ?? "the model"
        let voices = session.voices ?? []
        let makers = session.newVoiceChoices
        let making = fieldKey == nil || session.newVoice
        // A clip attached in Speak: the voice is the clip's, shown dimmed as "From clip".
        let fromClip = session.newVoice && session.form?.fields.contains { session.isCloneField($0) && !session.value($0.key).files.isEmpty } == true
        let selection =
            fromClip ? Self.fromClip : making ? Self.newVoicePrefix + (session.operation.flatMap { op in Self.voiceMakers(makers).contains(op) ? op : nil } ?? "") : current
        Picker(title, selection: Binding(get: { selection }, set: { choose($0) })) {
            Text("Default").tag("")
            if fromClip { Text("From clip").tag(Self.fromClip).selectionDisabled() }
            if !current.isEmpty, !voices.contains(where: { $0.id == current }) { Text(VoiceChoice.title(current, voices: voices)).tag(current) }
            if session.voices == nil {
                Section { Text("More voices once loaded").tag(Self.unlisted).selectionDisabled() }
            } else if voices.isEmpty {
                Section { Text("No other voices").tag(Self.unlisted).selectionDisabled() }
            }
            ForEach(Array(Self.groups(voices).enumerated()), id: \.offset) { _, group in
                if let heading = group.heading {
                    Section(heading) { ForEach(group.voices) { v in Text(v.name).tag(v.id) } }
                } else {
                    Section { ForEach(group.voices) { v in Text(v.name).tag(v.id) } }
                }
            }
            if !makers.isEmpty {
                Section {
                    ForEach(makers, id: \.title) { m in Text(m.title).tag(Self.newVoicePrefix + (m.operation ?? "")) }
                }
            }
        }
        .labelsHidden().pickerStyle(.menu).frame(maxWidth: .infinity, alignment: .leading).opacity(fromClip ? 0.6 : 1).controlID("field-\(fieldKey ?? "voice")")
        .windowHelp(session.voices == nil ? "Voices are listed once \(name) is loaded; a run loads it." : help)
    }

    private static func voiceMakers(_ makers: [(title: String, operation: String?)]) -> [String] { makers.compactMap(\.operation) }

    private func choose(_ tag: String) {
        if tag == Self.unlisted || tag == Self.fromClip { return }
        if tag.hasPrefix(Self.newVoicePrefix) {
            let op = String(tag.dropFirst(Self.newVoicePrefix.count))
            session.chooseNewVoice(op.isEmpty ? nil : op)
        } else if fieldKey == nil || session.newVoice {
            session.chooseVoice(tag)
        } else if let fieldKey {
            session.set(fieldKey, .text(tag))
        }
    }

    /// The voices in labelled groups: by language when they span more than one, and by sex where the ids carry it
    /// (Kokoro's "af_heart", "am_adam"): "Female", "Male", or "American English · Female". Never an unnamed gap.
    static func groups(_ voices: [VoiceChoice]) -> [(heading: String?, voices: [VoiceChoice])] {
        var byLanguage: [(heading: String?, voices: [VoiceChoice])] = []
        for row in VoiceChoice.rows(voices) {
            switch row {
            case .heading(let language): byLanguage.append((language, []))
            case .voice(let v):
                if byLanguage.isEmpty { byLanguage.append((nil, [])) }
                byLanguage[byLanguage.count - 1].voices.append(v)
            }
        }
        var out: [(heading: String?, voices: [VoiceChoice])] = []
        for group in byLanguage {
            let sexes = group.voices.map { sex($0) }
            guard sexes.allSatisfy({ $0 != nil }), Set(sexes.compactMap { $0 }).count > 1 else { out.append(group); continue }
            var order: [String] = [], split: [String: [VoiceChoice]] = [:]
            for v in group.voices {
                let s = sex(v) ?? ""
                if split[s] == nil { order.append(s) }
                split[s, default: []].append(v)
            }
            for s in order { out.append((group.heading.map { "\($0) \u{00b7} \(s)" } ?? s, split[s] ?? [])) }
        }
        return out
    }
    static func sex(_ v: VoiceChoice) -> String? {
        switch v.group {
        case "f": return "Female"
        case "m": return "Male"
        default: return nil
        }
    }
}
