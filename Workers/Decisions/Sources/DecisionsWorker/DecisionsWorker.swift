// SPDX-License-Identifier: MIT
// The decisions runner: Verdict's Laya and Von models over ArsenalWire (docs/CONTRACTS.md §4).
// One process per loaded model; unload = process exit. Request handling follows Verdict's helper
// (Projects/Verdict/Sources/VerdictHelper/main.swift, Service.swift, Registry.swift @ f8177ed, Apache-2.0).
// A library so Workers/Native links decisions with voices and chat into one binary with one MLX copy.
import Foundation
import Darwin
import MLX
import DecisionsProtocol
import VerdictEngine

public enum DecisionsWorker {
    public static func run() -> Never {
        // MLX enables TF32 GEMM by default; Von's f32 checkpoint drifts ~0.001 in probability under it, and MLX caches
        // the flag on first use. Verdict sets it before any model loads (Registry.swift); so do we, before MLX is touched.
        setenv("MLX_ENABLE_TF32", "0", 1)

        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        // Metal library: an explicit override, else a signed app's Contents/Resources/mlx.metallib (Verdict's layout);
        // else MLX finds mlx-swift_Cmlx.bundle next to the binary (build.sh output, the app's Helpers layout).
        if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] {
            setMetallib(URL(fileURLWithPath: path))
        } else {
            let bundled = executable.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Resources/mlx.metallib")
            if FileManager.default.fileExists(atPath: bundled.path) { setMetallib(bundled) }
        }

        // stdout carries frames only: keep a private descriptor for them and point fd 1 at stderr, so a stray print from
        // a library can never corrupt the stream.
        let frames = dup(1)
        dup2(2, 1)

        let worker = Worker(out: frames)
        worker.hello()
        do {
            while let payload = try Wire.read(0) {
                switch worker.handle(payload) {
                case .continue: continue
                case .exit: exit(0)
                case .malformed: throw Wire.Malformed(description: "a JSON frame that is not an object")
                }
            }
        } catch {
            FileHandle.standardError.write(Data("arsenal-worker-decisions: \(error)\n".utf8))
            exit(2)
        }
        exit(0)
    }
}
