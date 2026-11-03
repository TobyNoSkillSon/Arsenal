// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import ImageIO
import UniformTypeIdentifiers

let revision = 1
let badLoad = false

final class Fixture: @unchecked Sendable {
    let outputLock = NSLock()
    let stateLock = NSLock()
    var cancelled: Set<Int> = []
    let work = DispatchQueue(label: "fixture-work")
    func send(_ value: [String: Any]) {
        outputLock.lock(); defer { outputLock.unlock() }
        do {
            let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
            let length = UInt32(data.count + 1).bigEndian
            var packet = withUnsafeBytes(of: length) { Data($0) }; packet.append(1); packet.append(data)
            try FileHandle.standardOutput.write(contentsOf: packet)
        } catch { exit(1) }
    }
    func isCancelled(_ id: Int) -> Bool {
        stateLock.lock(); defer { stateLock.unlock() }; return cancelled.contains(id)
    }
    func cancel(_ id: Int) {
        stateLock.lock(); cancelled.insert(id); stateLock.unlock()
    }
    func memory() -> [String: Any] {
        let process = Process(), pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-o", "rss=", "-p", String(ProcessInfo.processInfo.processIdentifier)]
        process.standardOutput = pipe; process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return ["footprint_mb": 0, "mlx_active_mb": 0, "mlx_cache_mb": 0] }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let kb = Double(String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines))!
        return ["footprint_mb": kb * 1024 / 1_000_000, "mlx_active_mb": 0, "mlx_cache_mb": 0]
    }
    func perform(_ request: [String: Any]) {
        let id = request["id"] as? Int ?? 0
        func reply(_ result: [String: Any] = [:]) {
            send(["event": "status", "memory": memory()])
            send(result.merging(["id": id, "ok": true]) { _, new in new })
        }
        func failure(_ code: String, _ message: String) { send(["id": id, "ok": false, "error": ["code": code, "message": message]]) }
        switch request["op"] as? String {
        case "ping", "status": reply()
        case "load":
            if badLoad { failure("bad_revision", "Deterministic failed candidate load"); return }
            if revision > 1 { Thread.sleep(forTimeInterval: 0.1) }
            reply(["load_s": 0, "engine": "deterministic-swift-fixture"])
        case "unload": reply(); exit(0)
        case "generate":
            guard Set(request.keys).isSubset(of: ["id", "op", "value"]), let value = request["value"] as? Int else {
                failure("invalid_request", "unexpected compatibility parameter"); return
            }
            reply(["value": value])
        case "separate":
            guard let layers = request["layers"] as? Int, (1...8).contains(layers), let path = request["out_dir"] as? String else {
                failure("invalid_request", "layers and out_dir required"); return
            }
            let delay = request["delay"] as? Double ?? 0
            for _ in 0..<max(0, Int(delay * 100)) {
                if isCancelled(id) { failure("cancelled", "Cancelled"); return }; Thread.sleep(forTimeInterval: 0.01)
            }
            do {
                let directory = URL(fileURLWithPath: path)
                let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"]!).appendingPathComponent("jobs").resolvingSymlinksInPath().path + "/"
                guard directory.resolvingSymlinksInPath().path.hasPrefix(root) else { throw CocoaError(.fileWriteNoPermission) }
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                var files: [[String: Any]] = []
                for index in 0..<layers {
                    if isCancelled(id) { failure("cancelled", "Cancelled"); return }
                    send(["id": id, "event": "progress", "phase": "step", "label": "Separating layers", "unit": "layers", "step": index + 1, "total": layers])
                    let file = directory.appendingPathComponent("layer-\(index).png")
                    let pixels = Data([UInt8(index * 20), 80, 200, 255])
                    let provider = CGDataProvider(data: pixels as CFData)!
                    let image = CGImage(
                        width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
                    let destination = CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil)!
                    CGImageDestinationAddImage(destination, image, nil)
                    guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
                    let png = try Data(contentsOf: file)
                    let hash = SHA256.hash(data: png).map { String(format: "%02x", $0) }.joined()
                    files.append([
                        "bytes": png.count, "sha256": hash, "id": "layer-\(index)", "path": file.path, "mime": "image/png", "order": index, "label": "Layer \(index)",
                        "width": 1, "height": 1, "depth_min": Double(index) / Double(layers), "depth_max": Double(index + 1) / Double(layers)
                    ])
                }
                switch request["bad"] as? String {
                case "duplicate": files[layers - 1]["id"] = files[0]["id"]
                case "missing": files[layers - 1]["path"] = directory.appendingPathComponent("missing.png").path
                case "mime": files[layers - 1]["mime"] = "image/jpeg"
                case "count": files.removeLast()
                default: break
                }
                reply(["files": files, "metadata": ["revision": revision, "pid": ProcessInfo.processInfo.processIdentifier]])
            } catch { failure("fixture", error.localizedDescription) }
        default: failure("unsupported_op", "Unknown operation")
        }
    }
}
func exact(_ count: Int) throws -> Data? {
    var data = Data()
    while data.count < count {
        guard let part = try FileHandle.standardInput.read(upToCount: count - data.count), !part.isEmpty else {
            if data.isEmpty { return nil }; throw CocoaError(.fileReadCorruptFile)
        }
        data.append(part)
    }
    return data
}
let fixture = Fixture()
fixture.send([
    "event": "hello", "proto": 1, "worker": "fixture-layers", "runner": "swift", "pid": ProcessInfo.processInfo.processIdentifier,
    "build": "deterministic-fixture", "kinds": ["local.layers"], "ops": ["load", "ping", "status", "cancel", "unload", "separate", "generate"]
])
do {
    while let header = try exact(4) {
        let length = header.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        guard length > 0, length <= 16 * 1024 * 1024, let body = try exact(Int(length)), body.first == 1,
            let request = try JSONSerialization.jsonObject(with: body.dropFirst()) as? [String: Any]
        else { throw CocoaError(.fileReadCorruptFile) }
        if request["op"] as? String == "cancel" {
            let target = request["target"] as? Int ?? 0; fixture.cancel(target)
            fixture.send(["id": request["id"] as? Int ?? 0, "ok": true])
        } else {
            // Copy JSON bytes into the sendable closure; decoded Foundation containers stay on the worker queue.
            let bytes = Data(body.dropFirst())
            fixture.work.async { if let request = (try? JSONSerialization.jsonObject(with: bytes)) as? [String: Any] { fixture.perform(request) } }
        }
    }
    exit(0)
} catch { fputs("Invalid ArsenalWire frame\n", stderr); exit(1) }
