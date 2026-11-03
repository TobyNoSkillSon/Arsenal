// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import MCP

/// Fixed SSE bytes, served only by this test's URLSession: no global protocol registration or shared state.
private final class ResumeSSEProtocol: URLProtocol, @unchecked Sendable {
    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let url = request.url,
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "text/event-stream"])
        else { return }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("id: \(url.lastPathComponent)\ndata: {}\n\n".utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@Test func mcpNotificationResumeCursorNeverUsesAPostResponseEventID() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [ResumeSSEProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let endpoint = try #require(URL(string: "https://mcp-resume.invalid/mcp"))
    let transport = HTTPClientTransport(endpoint: endpoint, session: session)
    func consume(_ id: String, standalone: Bool = false) async throws {
        let (bytes, _) = try await session.bytes(from: endpoint.appendingPathComponent(id))
        #expect(try await transport.processSSE(bytes, standalone: standalone))
    }
    // Force the failing combined-run order: initialize POST finishes before notification GET starts.
    try await consume("initialize-post")
    #expect(await transport.lastEventID == nil)
    try await consume("notification-get", standalone: true)
    #expect(await transport.lastEventID == "notification-get")
    // Later POST responses must also leave the GET's reconnect cursor intact.
    try await consume("tool-result-post")
    #expect(await transport.lastEventID == "notification-get")
    try await consume("notification-get-reconnected", standalone: true)
    #expect(await transport.lastEventID == "notification-get-reconnected")
}
