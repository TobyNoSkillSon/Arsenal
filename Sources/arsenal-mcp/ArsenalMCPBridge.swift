// SPDX-License-Identifier: MIT
import Foundation
import Darwin

@main struct ArsenalMCPBridge {
    static func main() async {
        if CommandLine.arguments.contains("--help") {
            try? FileHandle.standardError.write(contentsOf: Data("arsenal-mcp: stdio bridge to the running Arsenal app; ARSENAL_SUPPORT_DIR selects an isolated instance.\n".utf8))
            return
        }
        do { try await run() } catch {
            try? FileHandle.standardError.write(contentsOf: Data("arsenal-mcp: \(error.localizedDescription)\n".utf8)); exit(1)
        }
    }
    static func run() async throws {
        let support = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] ?? NSHomeDirectory() + "/Library/Application Support/Arsenal")
        let statusURL = support.appendingPathComponent("status.json")
        func port() -> Int? {
            guard let data = try? Data(contentsOf: statusURL), let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
            guard let pid = object["pid"] as? Int32, kill(pid, 0) == 0 else { return nil }
            return object["port"] as? Int
        }
        if port() == nil, ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] == nil {
            let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            let app = executable.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            let launch = Process(); launch.executableURL = URL(fileURLWithPath: "/usr/bin/open"); launch.arguments = ["-g", app.path]; try launch.run(); launch.waitUntilExit()
            for _ in 0..<20 { if port() != nil { break }; try await Task.sleep(for: .milliseconds(250)) }
        }
        let bridge = Bridge(support: support)
        let lines = AsyncStream<String> { continuation in
            DispatchQueue(label: "arsenal.mcp.stdin").async {
                while let line = readLine() { continuation.yield(line) }; continuation.finish()
            }
        }
        for await line in lines { await bridge.accept(Data(line.utf8)) }
        await bridge.drain(); await bridge.close()
    }
}
