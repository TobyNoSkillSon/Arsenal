// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import ArsenalCore

public enum RunnerCheckCommand {
    public static func run(arguments args: [String]) -> Never {
        signal(SIGPIPE, SIG_IGN)
        do {
            if args.count == 2, args[0] == "--validate-output" {
                let request = try WireJSON.decode(Data(contentsOf: URL(fileURLWithPath: args[1])))
                guard let result = request["result"]?.object, let output = request["output"]?.string,
                    let name = request["operation"]?.string, let directory = request["directory"]?.string
                else { throw CheckFailure("Output validation needs result, operation, output and directory") }
                let operation = ModelOperation(name: name, title: name, inputSchema: [:], output: output, stream: false, outputSchema: request["schema"]?.object)
                _ = try OperationArtifacts.output(result, operation: operation, directory: URL(fileURLWithPath: directory))
                print("PASS production operation output"); exit(0)
            }
            if args.isEmpty || args == ["--help"] {
                print("Usage: arsenal-runner-check <cmd…> --model <folder> [--weights DIR] [--kind-dir DIR] [--timeout SECONDS] [--fast] [--results DIR]")
                exit(args.isEmpty ? 2 : 0)
            }
            guard let marker = args.firstIndex(of: "--model"), marker > 0, marker + 1 < args.count else { throw CheckFailure("use <runner command…> --model <folder>") }
            let command = Array(args[..<marker])
            let folder = URL(fileURLWithPath: args[marker + 1]).standardizedFileURL
            var weights: URL?; var kinds: URL?; var results: URL?
            var timeout = 30.0; var fast = false
            var i = marker + 2
            while i < args.count {
                if args[i] == "--fast" { fast = true; i += 1; continue }
                guard i + 1 < args.count else { throw CheckFailure("missing value for \(args[i])") }
                switch args[i] {
                case "--weights": weights = URL(fileURLWithPath: args[i + 1]).standardizedFileURL
                case "--kind-dir": kinds = URL(fileURLWithPath: args[i + 1]).standardizedFileURL
                case "--results": results = URL(fileURLWithPath: args[i + 1]).standardizedFileURL
                case "--timeout":
                    guard let value = Double(args[i + 1]), value.isFinite, value > 0, value <= 180 else { throw CheckFailure("timeout must be 0 < seconds <= 180") }
                    timeout = value
                default: throw CheckFailure("unknown option \(args[i])")
                }
                i += 2
            }
            let support = results ?? URL(fileURLWithPath: "/tmp/arsenal-runner-check/\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: support.appendingPathComponent("jobs"), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: support.appendingPathComponent("cache"), withIntermediateDirectories: true)
            let model = try CheckModel(folder: folder, kindDir: kinds ?? CheckModel.findKinds(near: folder), support: support)
            let checks = RunnerChecks(command: command, model: model, support: support, weights: weights ?? folder, timeout: timeout, fast: fast)
            do { try checks.run() } catch { checks.failed += 1; try checks.retainReport(); throw error }
            try checks.retainReport()
            print("\(checks.failed == 0 ? "PASS" : "FAIL") summary: \(checks.passed) passed, \(checks.failed) failed; logs: \(support.path)")
            exit(checks.failed == 0 ? 0 : 1)
        } catch { print("FAIL runner-check: \(String(describing: error).components(separatedBy: .newlines).joined(separator: " "))"); exit(1) }
    }
}
