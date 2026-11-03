// SPDX-License-Identifier: MIT
import Foundation
import Network
import ArsenalCore
import ArsenalBench

/// The app and test executable host the same listener and router. No separate inference service is started here.
public final class ArsenalAPIServer: @unchecked Sendable {
    public static let preferredPort: UInt16 = 8798
    private let router: APIRouter
    private let support: URL
    private let requestedPort: UInt16
    private let queue = DispatchQueue(label: "arsenal.http.listener")
    private let lock = NSLock()
    private var listener: NWListener?
    private var connections: [UUID: HTTPConnection] = [:]
    public private(set) var port = 0
    public init(service: any ModelsService, supportDirectory: URL? = nil, port: UInt16? = nil, benchmarkController: BenchmarkController? = nil) {
        support =
            supportDirectory
            ?? URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] ?? NSHomeDirectory() + "/Library/Application Support/Arsenal", isDirectory: true)
        requestedPort = port ?? ProcessInfo.processInfo.environment["ARSENAL_API_PORT"].flatMap(UInt16.init) ?? Self.preferredPort
        router = APIRouter(service: service, support: support, benchmarkController: benchmarkController)
    }
    @discardableResult public func start() async throws -> Int {
        try await router.prepareMCP()
        do { port = try await listen(requestedPort) } catch { guard requestedPort != 0 else { throw error }; port = try await listen(0) }
        await router.setPort(port)
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try jsonData(["api": 1, "port": port, "pid": ProcessInfo.processInfo.processIdentifier]).write(to: support.appendingPathComponent("status.json"), options: .atomic)
        return port
    }
    private func listen(_ requested: UInt16) async throws -> Int {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: requested) ?? .any)
        let candidate = try NWListener(using: parameters)
        lock.withLock { listener = candidate }
        candidate.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        return try await withCheckedThrowingContinuation { continuation in
            let settled = StartGate()
            candidate.stateUpdateHandler = { state in
                switch state {
                case .ready: if settled.once() { continuation.resume(returning: Int(candidate.port?.rawValue ?? 0)) }
                case .failed(let error): candidate.cancel(); if settled.once() { continuation.resume(throwing: error) }
                case .cancelled: if settled.once() { continuation.resume(throwing: CancellationError()) }
                default: break
                }
            }
            candidate.start(queue: queue)
        }
    }
    private final class StartGate: @unchecked Sendable {
        private let lock = NSLock(); private var used = false
        func once() -> Bool {
            lock.withLock {
                if used { return false }; used = true; return true
            }
        }
    }
    private func accept(_ connection: NWConnection) {
        let id = UUID()
        let session = HTTPConnection(connection) { [weak self] in _ = self?.lock.withLock { self?.connections.removeValue(forKey: id) } }
        let admitted = lock.withLock { () -> Bool in
            guard connections.count < 32 else { return false }; connections[id] = session; return true
        }
        guard admitted else { connection.cancel(); return }
        session.begin(port: port) { [router] request, session in try await router.handle(request, session) }
    }
    public func stop() {
        let active = lock.withLock { () -> [HTTPConnection] in
            listener?.cancel(); listener = nil; return Array(connections.values)
        }
        active.forEach { $0.close() }
        Task { await router.stopMCP() }
        let path = support.appendingPathComponent("status.json")
        if let data = try? Data(contentsOf: path), let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            object["pid"] as? Int == Int(ProcessInfo.processInfo.processIdentifier)
        {
            try? jsonData(["api": 1, "port": NSNull(), "pid": ProcessInfo.processInfo.processIdentifier]).write(to: path, options: .atomic)
        }
    }
}
