// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal
import ArsenalCore

@MainActor @Suite(.menuState, .serialized) struct Fix31Tests {
    @Test func menuImagesStayVisibleAndSwiftApprovalIsExplicit() {
        _ = NSApplication.shared
        let item = NSMenuItem(title: "Activity", action: nil, keyEquivalent: "")
        item.image = NSImage(systemSymbolName: "gauge", accessibilityDescription: nil)
        if #available(macOS 27.0, *) { item.setValue(1, forKey: "preferredImageVisibility") }
        item.showMenuImage(); #expect(item.image != nil)
        if #available(macOS 27.0, *) { #expect((item.value(forKey: "preferredImageVisibility") as? NSNumber)?.intValue == 1) }
        var row = LayersFixture.row; row.runnerLanguage = "swift"
        let alert = ModelsMenu.codeAlert(row)
        #expect(alert.informativeText.contains("Swift executable") && alert.informativeText.contains("your permissions"))
    }
    @Test func composedRootAndRequiredEmptyStringRemainSubmittable() throws {
        let schema: JSONObject = ["type": "object", "properties": ["base": ["type": "string"]], "allOf": [["required": ["count"], "properties": ["count": ["type": "integer"]]]]]
        let form = OperationForm(schema: schema); #expect(form.wholeJSON)
        let input = try form.input(["": .text(#"{"base":"", "count":1}"#)], stage: { $0.path })
        try OperationSchemaValidator.validate(.object(input), schema: .object(schema))
        let plain: JSONObject = ["type": "object", "required": ["base"], "properties": ["base": ["type": "string"]]]
        #expect(try OperationForm(schema: plain).input(["base": .text("")], stage: { $0.path }) == ["base": ""])
    }
    @Test func closedPreparationCanReopenAndRunningRevisionDefersRefresh() async throws {
        let session = OperationSession(model: ToolModel(LayersFixture.row), client: { nil }, inbox: FileManager.default.temporaryDirectory)
        session.prepare(); session.close(); session.reopen()
        for _ in 0..<20 { if !session.loading { break }; await Task.yield() }
        #expect(!session.loading && session.problem != nil)
        session.update(model: ToolModel(LayersFixture.row), blocked: nil, revision: "one")
        session.preview(operations: [LayersFixture.descriptor], run: ToolRun(model: LayersFixture.id, operation: "separate"))
        session.update(model: ToolModel(LayersFixture.row), blocked: nil, revision: "two")
        #expect(!session.loading)
        session.stop()
        #expect(session.loading)
        for _ in 0..<20 { if !session.loading { break }; await Task.yield() }
        #expect(session.problem != nil)
        session.discard()
    }
    @Test func failedValidationRemovesItsStagedInput() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("fix31-inbox-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("a.png"); try Data([1]).write(to: source)
        let session = OperationSession(model: ToolModel(LayersFixture.row), client: { nil }, inbox: root.appendingPathComponent("inbox"))
        session.preview(operations: [LayersFixture.descriptor], values: ["image": .files([source]), "layers": .text("bad")])
        session.start(); #expect(session.problem != nil)
        #expect((try FileManager.default.contentsOfDirectory(atPath: session.inbox.path)).isEmpty)
        session.discard()
    }
    @Test func speedFailureIsExplained() {
        var checks = ModelChecksStatus(protocolStatus: .pass, parity: .pass, task: .pass); checks.speed = .fail
        #expect(ModelHelp.checks(checks).contains("Speed failed"))
    }
    @Test func everyAppOwnedMenuImageAssignmentForcesVisiblePolicy() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let sources = root.appendingPathComponent("Sources/Arsenal")
        let files = try #require(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
        let assignment = try NSRegularExpression(pattern: #"\b(\w+)\.image\s*="#)
        var checked = 0
        for case let file as URL in files where file.pathExtension == "swift" {
            let source = try String(contentsOf: file, encoding: .utf8)
            let lines = source.components(separatedBy: "\n")
            for (index, line) in lines.enumerated() {
                for match in assignment.matches(in: line, range: NSRange(line.startIndex..., in: line)) {
                    let owner = String(line[try #require(Range(match.range(at: 1), in: line))])
                    // These controls bear images too; their declarations prove they are not menu items.
                    let control = try NSRegularExpression(pattern: "(?:let|var) " + owner + #"\s*(?:: NS(?:Button|ImageView))?\s*= NS(?:Button|ImageView)\("#)
                    if control.firstMatch(in: source, range: NSRange(source.startIndex..., in: source)) != nil { continue }
                    if line.contains(".button?.image") { continue }
                    if owner == "view", source.contains("guard let view = cell.imageView"), source.contains("if let image = content as? NSImageView { imageView = image }") {
                        continue
                    }
                    checked += 1
                    let following = lines[index...min(index + 3, lines.count - 1)].joined(separator: "\n")
                    #expect(following.contains(owner + ".showMenuImage()"), "\(file.lastPathComponent):\(index + 1) overrides \(owner)'s visible image policy")
                }
            }
        }
        #expect(checked > 0, "The menu image scan must inspect assignments")
        let quit = try #require(LaunchDelegate.startupMenu().items.last)
        #expect(quit.image != nil && quit.menuImageIsVisible)
    }
    @Test func anUnloadedToolsFormRefreshesWhenItsApprovedDescriptorChanges() async throws {
        _ = NSApplication.shared
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("fix31-schema-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        var row = LayersFixture.row
        row.states = ["bf16": .init(state: .ready)]; row.revision = .init()
        row.approval = .init(by: .app, digest: "old")
        let old = ModelOperation(
            name: "work", title: "Work", inputSchema: ["type": "object", "properties": ["old": ["type": "string"]]], output: "json", stream: false, schemaVersion: 3)
        let new = ModelOperation(
            name: "work", title: "Work", inputSchema: ["type": "object", "properties": ["new": ["type": "integer"]]], output: "json", stream: false, schemaVersion: 3)
        let oldAPI = APIHost(service: StubModelsService(rows: [row], operations: [row.id: [old]]), support: support.appendingPathComponent("old"), port: 0)
        let newAPI = APIHost(service: StubModelsService(rows: [row], operations: [row.id: [new]]), support: support.appendingPathComponent("new"), port: 0)
        await oldAPI.start(); await newAPI.start()
        defer { oldAPI.stop(); newAPI.stop() }
        var client = ArsenalClient(port: try #require(oldAPI.port))
        let center = ToolCenter(client: { client }, playback: FixturePlayback())
        center.present = { _ in }; center.inbox = support.appendingPathComponent("inbox")
        center.open(.operation, row)
        defer { center.closeAll() }
        let session = try #require(center.operations[row.id]?.session)
        for _ in 0..<100 { if !session.loading { break }; try await Task.sleep(for: .milliseconds(10)) }
        #expect(session.form?.fields.map(\.key) == ["old"])
        client = ArsenalClient(port: try #require(newAPI.port))
        row.approval = .init(by: .app, digest: "new")
        #expect(row.revision?.active == nil && row.residentDType == nil)
        center.update(rows: [row])
        #expect(session.loading)
        for _ in 0..<100 { if !session.loading { break }; try await Task.sleep(for: .milliseconds(10)) }
        #expect(session.form?.fields.map(\.key) == ["new"])
        #expect(session.problem == nil && !session.loading)
        center.update(rows: [row]); #expect(!session.loading)
    }

}
