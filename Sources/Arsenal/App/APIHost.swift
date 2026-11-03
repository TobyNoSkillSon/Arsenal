// SPDX-License-Identifier: MIT
import Foundation
import ArsenalAPI
import ArsenalCore

/// The app hosts the API listener (mb-api's `ArsenalAPIServer`) over the same service as the table, so the menu, the
/// API and MCP clients see one set of loaded models. The server writes `status.json` (port + pid) in the support dir.
@MainActor final class APIHost {
    private let server: ArsenalAPIServer
    private(set) var port: Int?
    private(set) var error: String?

    /// `port` nil: `ARSENAL_API_PORT` or the preferred port (8798), falling back to any free one.
    init(service: any ModelsService, support: URL, port: UInt16? = nil) {
        server = ArsenalAPIServer(service: service, supportDirectory: support, port: port)
    }

    func start() async {
        do { port = try await server.start(); error = nil } catch {
            self.error = "The API could not start: \(error.localizedDescription)"
        }
    }
    func stop() { server.stop(); port = nil }
}
