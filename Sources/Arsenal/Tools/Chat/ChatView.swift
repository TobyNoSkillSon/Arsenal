// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI
import UniformTypeIdentifiers

// The chat test window: the model on the traffic lights' line, the live figures, the speed presets, the conversation,
// the message field. No tools, no system prompt of its own: the model's own settings apply. Pictures and clips attach
// (paperclip, drop, ⌘V) and an answer can be held to named fields (a JSON Schema) only when the model's chat schema
// declares those content parts and response format.

struct ChatView: View {
    @ObservedObject var session: ChatSession
    @Environment(\.palette) private var p
    @FocusState private var focused: Bool
    @ToolDetailsShown(ToolDetails.key) private var details = false
    static let size = NSSize(width: 880, height: 720), minSize = NSSize(width: 860, height: 480)
    static let inset: CGFloat = 20

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ToolBand(detail: "") {
                TaskTitle(title: "Chat") {
                    if session.choices.count > 1 { ChatModelPicker(session: session).controlID("model") } else { ModelName(model: session.model) }
                }
            } trailing: {
                if session.capabilities.jsonSchema {
                    BandButton(
                        symbol: "curlybraces", help: session.structured ? "Answer in plain text" : "Answer with named fields (JSON)", enabled: !session.generating,
                        on: session.structured
                    ) {
                        session.structured.toggle()
                    }
                    .controlID("structured")
                }
                DetailsButton()
                BandButton(symbol: "square.and.pencil", help: "New chat", enabled: !session.messages.isEmpty) { session.newChat() }.controlID("new-chat")
            }
            // The figures and the benchmark prompts are lab details: off until Details shows them.
            if details {
                // No reference measured: no Reference rows (a label beside nothing reads as broken).
                TelemetryGrid(figures: session.figures(), reference: session.referenceLine, showsReference: session.referenceLine != nil)
                    .padding(.horizontal, Self.inset).padding(.top, 6).padding(.bottom, 14)
            }
            Rectangle().fill(p.hairline.swiftUI).frame(height: 1)
            if details { presets.padding(.horizontal, Self.inset).padding(.vertical, 10) }
            if let notice = session.notice {
                Text(notice).font(.system(size: 12)).foregroundStyle(p.red.swiftUI).lineLimit(2).padding(.horizontal, Self.inset).padding(.bottom, 6)
            }
            transcript
            if session.structured, session.capabilities.jsonSchema {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Fields of the answer").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI)
                    SchemaEditor(key: "response", text: $session.responseSchema).disabled(session.generating)
                }
                .padding(.horizontal, Self.inset).padding(.top, 6)
            }
            composer.padding(.horizontal, Self.inset - 4).padding(.bottom, 14).padding(.top, 6)
        }
        .onAppear { focused = true }
        // A benchmark prompt waiting to be sent goes with the Details that offered it.
        .onChange(of: details) { _, shown in if !shown { session.preset = nil } }
    }

    private var presets: some View {
        HStack(spacing: 6) {
            ForEach(PresetPrompt.all) { preset in
                let reason = session.unavailable(preset)
                Pill(
                    title: preset.title, on: session.preset == preset, enabled: reason == nil && !session.generating,
                    help: reason ?? "Insert a \(preset.tokens.grouped)-token prompt"
                ) { session.insert(preset) }.controlID("preset-\(preset.title)")
            }
            Spacer(minLength: 0)
        }
    }

    private var transcript: some View {
        ScrollView(.vertical) {
            LazyVStack(alignment: .leading, spacing: 14) {
                ForEach(session.messages) { m in
                    MessageView(message: m).id(m.id)
                }
            }.padding(.horizontal, Self.inset).padding(.vertical, 12)
        }
        .defaultScrollAnchor(.bottom)
        .scrollIndicators(.automatic)
        .frame(maxHeight: .infinity)
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if session.capabilities.attaches {
                Button {
                    session.pickAttachments()
                } label: {
                    Image(systemName: "paperclip").font(.system(size: 14)).frame(width: 26, height: 26).contentShape(Rectangle())
                }
                .buttonStyle(.plain).foregroundStyle(p.t2.swiftUI).disabled(session.generating)
                .windowHelp(Self.attachHelp(session.capabilities)).accessibilityLabel("Attach").controlID("attach")
                .padding(.leading, 6).padding(.bottom, 6)
            }
            VStack(alignment: .leading, spacing: 6) {
                if !session.attachments.isEmpty {
                    ScrollView(.horizontal) {
                        HStack(spacing: 6) { ForEach(session.attachments, id: \.self) { AttachmentChip(url: $0, remove: { session.detach($0) }) } }
                    }.scrollIndicators(.never)
                }
                if let preset = session.preset {
                    HStack(spacing: 5) {
                        Image(systemName: "text.alignleft").font(.system(size: 10.5, weight: .semibold))
                        Text("\(preset.title) prompt").font(.system(size: 12, weight: .medium))
                        Button {
                            session.preset = nil
                        } label: {
                            Image(systemName: "xmark").font(.system(size: 9, weight: .bold)).frame(width: 16, height: 16).contentShape(Rectangle())
                        }.buttonStyle(.plain).windowHelp("Remove the prompt").controlID("remove-preset")
                    }
                    .foregroundStyle(p.accentText.swiftUI).padding(.leading, 8).padding(.trailing, 3).frame(height: 22)
                    .background(p.accent.withAlphaComponent(0.2).swiftUI, in: RoundedRectangle(cornerRadius: 6))
                }
                TextField("Message \(session.model.name)", text: $session.draft, axis: .vertical)
                    .textFieldStyle(.plain).font(.system(size: 13.5)).lineLimit(1...8)
                    .focused($focused).controlID("message")
                    .onSubmit { session.send() }
            }
            .padding(.vertical, 9).padding(.leading, session.capabilities.attaches ? 0 : 12)
            Group {
                if session.generating {
                    RoundButton(symbol: "stop.fill", help: "Stop", enabled: true, filled: true) { session.stop() }.controlID("stop")
                } else {
                    RoundButton(symbol: "arrow.up", help: "Send", enabled: session.canSend, filled: true) { session.send() }.controlID("send")
                }
            }.padding(.trailing, 6).padding(.bottom, 6)
        }
        .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(p.raisedBorder.swiftUI, lineWidth: 1))
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            guard session.capabilities.attaches, !session.generating else { return false }
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in session.attach([url]) }
                }
            }
            return true
        }
    }

    /// "Attach pictures or WAV clips".
    static func attachHelp(_ c: ChatCapabilities) -> String {
        let kinds = [c.images ? "pictures" : nil, c.audioFormats.isEmpty ? nil : c.audioFormats.map { $0.uppercased() }.joined(separator: " or ") + " clips"].compactMap { $0 }
        return "Attach " + kinds.joined(separator: " or ")
    }
}

/// An attachment waiting in the composer: its thumbnail (or a clip's name) and ×.
struct AttachmentChip: View {
    let url: URL
    /// Nil in a sent message: it can no longer be removed.
    var remove: ((URL) -> Void)?
    @Environment(\.palette) private var p
    var body: some View {
        ZStack(alignment: .topTrailing) {
            Group {
                if ToolFiles.isImageFile(url) {
                    Thumbnail(url: url).frame(width: 44, height: 44).clipShape(RoundedRectangle(cornerRadius: 6))
                } else {
                    Label(url.lastPathComponent, systemImage: "waveform").font(.system(size: 11.5)).lineLimit(1).padding(.horizontal, 8).frame(height: 44)
                        .background(p.raised.swiftUI, in: RoundedRectangle(cornerRadius: 6))
                }
            }.accessibilityLabel(url.lastPathComponent)
            if let remove {
                Button {
                    remove(url)
                } label: {
                    Image(systemName: "xmark.circle.fill").font(.system(size: 12)).symbolRenderingMode(.palette).foregroundStyle(Color.white, Color.black.opacity(0.6))
                        .frame(width: 18, height: 18).contentShape(Rectangle())
                }.buttonStyle(.plain).windowHelp("Remove").accessibilityLabel("Remove \(url.lastPathComponent)").controlID("remove-attachment")
            }
        }
    }
}

struct MessageView: View {
    let message: ChatMessage
    @Environment(\.palette) private var p

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer(minLength: 80)
                VStack(alignment: .trailing, spacing: 6) {
                    if let preset = message.preset {
                        Label("\(preset.title) prompt", systemImage: "text.alignleft").font(.system(size: 12, weight: .medium))
                            .foregroundStyle(p.accentText.swiftUI)
                    }
                    if !message.attachments.isEmpty {
                        HStack(spacing: 6) { ForEach(message.attachments, id: \.self) { AttachmentChip(url: $0) } }
                    }
                    if !message.text.isEmpty {
                        Text(message.text).font(.system(size: 13.5)).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(p.selFill.swiftUI, in: RoundedRectangle(cornerRadius: 12))
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 6) {
                if !message.reasoning.isEmpty {
                    Text(message.reasoning).font(.system(size: 12)).foregroundStyle(p.t3.swiftUI).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if message.structured, !message.streaming, let data = message.text.data(using: .utf8),
                    let value = try? JSONDecoder().decode(JSONValue.self, from: data)
                {
                    Text(JSONView.styled(value, palette: p)).font(.system(size: 12.5, design: .monospaced)).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true).padding(10).frame(maxWidth: .infinity, alignment: .leading)
                        .background(p.raised.swiftUI, in: RoundedRectangle(cornerRadius: 8))
                } else if !message.text.isEmpty {
                    Text(message.text).font(.system(size: 13.5)).lineSpacing(2).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                } else if message.streaming, message.reasoning.isEmpty {
                    Text("\u{2026}").font(.system(size: 13.5)).foregroundStyle(p.t3.swiftUI)
                }
                if let error = message.error {
                    Label(error, systemImage: "exclamationmark.triangle.fill").font(.system(size: 12)).foregroundStyle(p.red.swiftUI)
                        .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: Chat controls

/// The send/stop circle (26 pt).
struct RoundButton: View {
    let symbol: String
    let help: String
    var enabled = true
    var filled = true
    let action: () -> Void
    @Environment(\.palette) private var p
    var body: some View {
        Button(action: action) {
            Image(systemName: symbol).font(.system(size: 12, weight: .bold)).foregroundStyle(enabled ? Color.white : p.t3.swiftUI)
                .frame(width: 26, height: 26)
                .background((enabled ? p.accent : p.selFill).swiftUI, in: Circle())
                .contentShape(Circle())
        }
        .buttonStyle(.plain).disabled(!enabled).windowHelp(help).accessibilityLabel(help)
    }
}

/// Chat's model pop-up, drawn as every window's (ModelPicker): the model's name in the secondary colour with its chevron;
/// choosing another model shows that model's own conversation.
struct ChatModelPicker: View {
    @ObservedObject var session: ChatSession
    @Environment(\.palette) private var p
    var body: some View {
        Menu {
            ForEach(session.choices) { choice in
                Button {
                    session.switchModel(choice.id)
                } label: {
                    if choice.id == session.model.id { Label(ModelPicker.title(choice), systemImage: "checkmark") } else { Text(ModelPicker.title(choice)) }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Text(session.model.name).font(.system(size: 13)).foregroundStyle(p.t2.swiftUI).lineLimit(1)
                Image(systemName: "chevron.down").font(.system(size: 9, weight: .semibold)).foregroundStyle(p.t3.swiftUI)
            }
            .padding(.vertical, 4).contentShape(Rectangle())
        }
        .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden).fixedSize()
        .disabled(session.generating)
        .windowHelp(session.generating ? "The model can change once this answer is done." : ModelPicker.precisionLine(session.model))
        .accessibilityLabel("Model: \(session.model.name)")
    }
}
