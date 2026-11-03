// SPDX-License-Identifier: MIT
import SwiftUI

// Alternatives a capability window offers as one chooser: options of which a run takes one (a scale or a height,
// labels or a phrase or an instruction, text or a file). The chooser is a row of pills; only the chosen option's
// controls show, and only its values are sent. Every declared option stays one click away.

/// Options of which a run takes one.
struct OptionGroup: Equatable, Identifiable {
    struct Option: Equatable {
        var title: String
        var keys: [String]
    }
    var id: String
    var options: [Option]
}

extension CapabilitySession {
    /// The groups the chosen operation offers, limited to options whose inputs the model declares (one left is no
    /// choice: its controls show as usual).
    var optionGroups: [OptionGroup] {
        guard let op = descriptor, let form else { return [] }
        let keys = Set(form.fields.map(\.key))
        return groups(op).compactMap { group in
            var g = group
            g.options = group.options.compactMap { o in
                let present = o.keys.filter(keys.contains)
                return present.isEmpty ? nil : OptionGroup.Option(title: o.title, keys: present)
            }
            return g.options.count > 1 ? g : nil
        }
    }

    private func choiceKey(_ group: OptionGroup) -> String { "\(operation ?? "")/\(group.id)" }
    /// The chosen option: as chosen, else the first one holding a value, else the first.
    func chosenIndex(_ group: OptionGroup) -> Int {
        if let i = chosen[choiceKey(group)], group.options.indices.contains(i) { return i }
        return group.options.firstIndex { $0.keys.contains { valueHeld($0) } } ?? 0
    }
    func choose(_ group: OptionGroup, _ index: Int) {
        guard !running else { return }
        chosen[choiceKey(group)] = index
        if let field = problem?.field, group.options.flatMap(\.keys).contains(field) { problem = nil }
    }
    /// A key of an option not chosen: not shown, not sent.
    func hidden(_ key: String) -> Bool {
        if hiddenOutsideGroups(key) { return true }
        for group in optionGroups {
            let chosen = chosenIndex(group)
            for (i, option) in group.options.enumerated() where i != chosen && option.keys.contains(key) { return true }
        }
        return false
    }
    /// Keys drawn inside a chooser rather than among the options.
    var groupedKeys: Set<String> { Set(optionGroups.flatMap { $0.options.flatMap(\.keys) }) }

    /// A value is there, whatever its control (not `filled`, which also asks whether it is hidden).
    func valueHeld(_ key: String) -> Bool {
        switch values[key] {
        case .files(let f)?: return !f.isEmpty
        case .text(let t)?: return !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .toggle(let on)?: return on
        case .choice(let i)?: return i != nil
        case nil: return false
        }
    }
}

/// One chooser: its pills, then the chosen option's controls (the main text as the main editor, files in the window's
/// well, the rest as their own controls without a second title).
struct OptionGroupView: View {
    let group: OptionGroup
    @ObservedObject var session: CapabilitySession
    @Environment(\.palette) private var p

    var body: some View {
        let chosen = session.chosenIndex(group)
        let subjects = Set(session.subjectFields.map(\.key))
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                ForEach(Array(group.options.enumerated()), id: \.offset) { i, option in
                    Pill(title: option.title, on: i == chosen, enabled: !session.running, help: option.title) { session.choose(group, i) }
                        .controlID("choose-\(group.id)-\(i)")
                }
            }
            ForEach(session.form?.fields.filter { group.options[chosen].keys.contains($0.key) && !subjects.contains($0.key) } ?? []) { field in
                if field.isMain {
                    MainEditor(field: field, session: session)
                } else if field.compact {
                    FieldControl(field: field, session: session).frame(maxWidth: .infinity, alignment: .leading)
                    FieldProblem(session: session, keys: [field.key])
                } else {
                    FieldControl(field: field, session: session, titled: false)
                }
            }
        }
        .id(group.id)
    }
}

/// The choosers of a capability window, in their place: those that choose what the run works on lead the column (a
/// text or a file), the rest head the options.
struct OptionGroupsView: View {
    @ObservedObject var session: CapabilitySession
    enum Place { case beforeSubject, afterSubject, options }
    let place: Place
    var body: some View {
        let subjects = Set(
            session.form?.fields.filter {
                if case .generic(.file) = $0.control { return true }; return false
            }.map(\.key) ?? [])
        let main = session.form?.main?.key
        let groups = session.optionGroups.filter { g in
            let keys = Set(g.options.flatMap(\.keys))
            let touchesSubject = !keys.isDisjoint(with: subjects)
            let touchesMain = main.map(keys.contains) ?? false
            switch place {
            case .beforeSubject: return touchesSubject
            case .afterSubject: return touchesMain && !touchesSubject
            case .options: return !touchesSubject && !touchesMain
            }
        }
        ForEach(groups) { OptionGroupView(group: $0, session: session) }
    }
}
