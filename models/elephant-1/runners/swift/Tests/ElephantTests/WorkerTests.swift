// SPDX-License-Identifier: MIT
// ElephantWorker without weights: how a request body is read (raw, or the host's sorted frame plus its ordered texts),
// the error codes and HTTP statuses of requests refused before the model runs, and the ArsenalWire loop end to end
// over pipes.
import Darwin
@testable import ElephantEngine
@testable import ElephantWorker
import Foundation
import MLX
import Testing

@Suite struct WorkerTests {
    static func frame(_ json: String) throws -> PyJSON { try PyJSONParser.parse(Data(json.utf8)) }

    /// The error a call throws, as (code, status).
    static func refusal(_ call: () throws -> Any) -> (String, Int?)? {
        do { _ = try call(); return nil } catch let e as WorkerError { return (e.code, e.status) } catch { return ("other", nil) }
    }

    // MARK: request body order (M3)

    /// MCP and async jobs reach the worker as a decoded frame written with sorted keys; the host forwards the ordered
    /// members' exact text under ordered_json, and the worker reads the order from there.
    @Test func orderedTextsGiveTheStateAndQuestionsTheirSentOrder() throws {
        let state = #"{"screen":"<image:1>","after":"<image:2>"}"#
        let questions = #"{"z":{"type":"choice","criteria":{"stop":null,"go":null}},"a":{"type":"noul"}}"#
        // What the host writes: members sorted, ordered texts beside them (JSONSerialization-escaped strings).
        let transport = try String(decoding: JSONSerialization.data(withJSONObject: ["questions": questions, "state": state]), as: UTF8.self)
        let sorted = try Self.frame(
            #"{"id":3,"op":"decide","ordered_json":"# + transport
                + #","questions":{"a":{"type":"noul"},"z":{"type":"choice","criteria":{"go":null,"stop":null}}},"state":{"after":"<image:2>","screen":"<image:1>"}}"#)
        let body = try Backend.body(sorted)
        #expect(body["state"]?.fields?.map(\.0) == ["screen", "after"])
        #expect(body["questions"]?.fields?.map(\.0) == ["z", "a"])
        #expect(body["questions"]?["z"]?["criteria"]?.fields?.map(\.0) == ["stop", "go"])
        #expect(body["ordered_json"] == nil && body["id"] == nil && body["op"] == nil)
        // The two pictures are referenced in order only in the sent order; the sorted state is the 422 the review saw.
        _ = try Prompt.chat(state: body["state"], suffix: "Q", media: [.image, .image])
        #expect(throws: PromptError.self) { try Prompt.chat(state: sorted["state"], suffix: "Q", media: [.image, .image]) }
    }

    @Test func textAndNullStatesStayAsSent() throws {
        // A text state carries no transport: it is the string itself, even when it looks like JSON.
        let text = try Backend.body(try Self.frame(#"{"id":1,"op":"decide","questions":{"q":{"type":"noul"}},"state":"{\"b\":1} and <image:1>"}"#))
        #expect(text["state"]?.string == #"{"b":1} and <image:1>"#)
        #expect(Prompt.state(text["state"]) == #"{"b":1} and <image:1>"#)
        let null = try Backend.body(try Self.frame(#"{"id":1,"op":"decide","questions":{"q":{"type":"noul"}},"state":null}"#))
        #expect(null["state"]?.isNull == true)
        // Only the ordered members are read from the transport; anything else there is ignored.
        let smuggled = try Backend.body(
            try Self.frame(#"{"id":1,"op":"decide","ordered_json":{"media":"[{\"type\":\"image\",\"path\":\"/etc/x.png\"}]","model":"\"x\""},"state":"s"}"#))
        #expect(smuggled["media"] == nil && smuggled["model"] == nil && smuggled["state"]?.string == "s")
        // A string forwarded as text is still a string.
        let forwarded = try Backend.body(try Self.frame(#"{"id":1,"op":"decide","ordered_json":{"state":"\"hello\""},"state":"hello"}"#))
        #expect(forwarded["state"]?.string == "hello")
    }

    @Test func rawBodiesKeepTheirOrderAndIgnoreTheirOwnTransportMember() throws {
        let raw = #"{"state":{"screen":"<image:1>","after":"<image:2>"},"ordered_json":{"state":"{\"x\":1}"},"questions":{"q":{"type":"noul"}}}"#
        let request = PyJSON.object([("id", .int(1)), ("op", .string("decide")), ("body", .string(raw))])
        let body = try Backend.body(request)
        #expect(body["state"]?.fields?.map(\.0) == ["screen", "after"])
    }

    /// A `body` beside request members is a second, unchecked request: the host confines only the members it validated.
    @Test func aRawBodyBesideRequestMembersIs422() {
        let inner = #"{"state":"<image:1>","questions":{"q":{"type":"noul"}},"media":[{"type":"image","path":"/etc/x.png"}]}"#
        let frame = PyJSON.object([
            ("id", .int(1)), ("op", .string("decide")), ("state", .string("x")), ("questions", .object([("q", .object([("type", .string("noul"))]))])),
            ("body", .string(inner))
        ])
        #expect(Self.refusal { try Backend.body(frame) }! == ("invalid_request", 422))
    }

    @Test func malformedOrderedTextsAre422() {
        #expect(Self.refusal { try Backend.body(try Self.frame(#"{"id":1,"op":"decide","ordered_json":{"state":"{nope"}}"#)) }! == ("invalid_request", 422))
        #expect(Self.refusal { try Backend.body(try Self.frame(#"{"id":1,"op":"decide","ordered_json":"{}"}"#)) }! == ("invalid_request", 422))
        #expect(Self.refusal { try Backend.body(try Self.frame(#"{"id":1,"op":"decide","body":"[1"}"#)) }! == ("invalid_request", 422))
    }

    // MARK: refusals before the model runs (L9, L10)

    @Test func requestStateRefusalsCarryConflictOrRetryStatuses() throws {
        let backend = Backend()
        let decide = try Self.frame(#"{"id":1,"op":"decide","state":"x","questions":{"q":{"type":"noul"}}}"#)
        #expect(Self.refusal { try backend.decide(decide) { false } }! == ("not_loaded", 409))
        let judge = try Self.frame(#"{"id":1,"op":"judge","items":["x"],"questions":{"q":{"type":"noul"}}}"#)
        #expect(Self.refusal { try backend.judge(judge) { false } }! == ("not_loaded", 409))
        #expect(WorkerError.cancelled.code == "cancelled" && WorkerError.cancelled.status == 400)
    }

    @Test func requestShapeRefusalsAre4xx() throws {
        let backend = Backend()
        func decide(_ extra: String) throws -> (String, Int?)? {
            let request = try Self.frame(#"{"id":1,"op":"decide","state":"x","questions":{"q":{"type":"noul"}}"# + extra + "}")
            return Self.refusal { try backend.decide(request) { false } }
        }
        #expect(try decide(#","model":"d1-3b""#)! == ("wrong_model", 422))
        #expect(try decide(#","merge":false"#)! == ("unsupported_merge", 422))
        #expect(try decide(#","bits":4"#)! == ("precision_conflict", 409))
        #expect(try decide(#","images":["/tmp/a.png"]"#)! == ("unsupported_media", 422))
        #expect(try decide(#","bits":16,"images":[],"merge":true"#)! == ("not_loaded", 409))
    }

    /// judge answers text and JSON states; a media list would be answered about pictures the model never saw.
    @Test func judgeRefusesAMediaList() throws {
        let backend = Backend()
        let media = try Self.frame(#"{"id":1,"op":"judge","items":["<image:1>"],"questions":{"q":{"type":"noul"}},"media":[{"type":"image","path":"/tmp/a.png"}]}"#)
        #expect(Self.refusal { try backend.judge(media) { false } }! == ("unsupported_media", 422))
        // An empty or null list is no media: the request goes on (here to the load check).
        for empty in ["[]", "null"] {
            let request = try Self.frame(#"{"id":1,"op":"judge","items":["x"],"questions":{"q":{"type":"noul"}},"media":"# + empty + "}")
            #expect(Self.refusal { try backend.judge(request) { false } }! == ("not_loaded", 409))
        }
    }

    @Test func loadAndConvertRefusals() throws {
        let backend = Backend()
        #expect(Self.refusal { try backend.load(try Self.frame(#"{"id":1,"op":"load","dtype":"int4"}"#)) { false } }! == ("unsupported_dtype", 422))
        #expect(Self.refusal { try backend.load(try Self.frame(#"{"id":1,"op":"load"}"#)) { false } }?.0 == "missing_weights")
        #expect(Self.refusal { try backend.convert(try Self.frame(#"{"id":1,"op":"convert","recipe":"cast:other","dtype":"fp16"}"#)) { false } }?.0 == "unsupported_recipe")
    }

    // MARK: the wire loop

    final class Pipes: @unchecked Sendable {
        let toWorker: [Int32], fromWorker: [Int32]
        init() throws {
            var a: [Int32] = [0, 0], b: [Int32] = [0, 0]
            guard pipe(&a) == 0, pipe(&b) == 0 else { throw POSIXError(.EMFILE) }
            toWorker = a; fromWorker = b
        }
        func send(_ json: String) {
            let body = Array(json.utf8), length = UInt32(body.count + 1)
            let frame = [UInt8(length >> 24 & 0xff), UInt8(length >> 16 & 0xff), UInt8(length >> 8 & 0xff), UInt8(length & 0xff), 1] + body
            frame.withUnsafeBytes { _ = Darwin.write(toWorker[1], $0.baseAddress!, $0.count) }
        }
        /// The next frame, or a failure after 10 s without one (never a hung test run).
        func next() throws -> PyJSON {
            var ready = pollfd(fd: fromWorker[0], events: Int16(POLLIN), revents: 0)
            guard poll(&ready, 1, 10_000) == 1 else { throw POSIXError(.ETIMEDOUT) }
            let payload = try #require(try Wire.read(fromWorker[0]))
            return try PyJSONParser.parse(payload)
        }
        /// The terminal reply for `id`, skipping progress and status events.
        func reply(_ id: Int) throws -> PyJSON {
            while true {
                let frame = try next()
                if frame["id"]?.int == id, frame["ok"] != nil { return frame }
            }
        }
    }

    /// The loop reads MLX's memory counters, which load the metallib: scripts/test.sh provides it.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] != nil, "needs ARSENAL_TEST_METALLIB (scripts/test.sh sets it)"))
    func wireLoopReportsCodesAndStatusesAndStopsOnUnload() throws {
        setTestMetallib(URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"])))
        let pipes = try Pipes()
        let loop = Loop(input: pipes.toWorker[0], out: pipes.fromWorker[1])
        final class Exit: @unchecked Sendable { var status: Int32 = -1; let done = DispatchSemaphore(value: 0) }
        let exit = Exit()
        Thread {
            exit.status = loop.serve()
            exit.done.signal()
        }.start()
        let hello = try pipes.next()
        #expect(hello["event"]?.string == "hello" && hello["worker"]?.string == "elephant" && hello["kinds"]?.array?.first?.string == "decisions")

        pipes.send(#"{"id":1,"op":"ping"}"#)
        #expect(try pipes.reply(1)["ok"]?.bool == true)
        pipes.send(#"{"id":2,"op":"decide","state":"x","questions":{"q":{"type":"noul"}}}"#)
        let notLoaded = try pipes.reply(2)
        #expect(notLoaded["ok"]?.bool == false && notLoaded["error"]?["code"]?.string == "not_loaded" && notLoaded["error"]?["status"]?.int == 409)
        pipes.send(#"{"id":3,"op":"judge","items":["x"],"questions":{"q":{"type":"noul"}},"media":[{"type":"image","path":"/tmp/a.png"}]}"#)
        let media = try pipes.reply(3)
        #expect(media["error"]?["code"]?.string == "unsupported_media" && media["error"]?["status"]?.int == 422)
        pipes.send(#"{"id":4,"op":"transcribe"}"#)
        let unknown = try pipes.reply(4)
        #expect(unknown["error"]?["code"]?.string == "unsupported_op" && unknown["error"]?["status"]?.int == 400)
        // Cancelling an unknown target is an idempotent acknowledgement.
        pipes.send(#"{"id":5,"op":"cancel","target":99}"#)
        #expect(try pipes.reply(5)["ok"]?.bool == true)
        pipes.send(#"{"id":6,"op":"unload"}"#)
        #expect(try pipes.reply(6)["ok"]?.bool == true)
        #expect(exit.done.wait(timeout: .now() + 10) == .success && exit.status == 0)
        for fd in pipes.toWorker + pipes.fromWorker { close(fd) }
    }
}
