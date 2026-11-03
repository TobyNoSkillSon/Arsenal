// SPDX-License-Identifier: MIT
import Foundation

enum LocalSwiftRunner {
    static func validate(_ runner: RunnerSpec, folder: URL) throws {
        guard let entry = runner.entry, entry.hasPrefix("runners/swift/bin/") else {
            throw ManifestError.invalid("Swift entry must be inside runners/swift/bin/")
        }
        let binary = try SafePath.resolve(entry, inside: folder)
        let attributes = try FileManager.default.attributesOfItem(atPath: binary.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular, FileManager.default.isExecutableFile(atPath: binary.path) else {
            throw ManifestError.invalid("Swift entry must be a built regular executable")
        }
        let package = try SafePath.resolve("runners/swift/Package.swift", inside: folder)
        guard try package.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true,
            try SafePath.files(in: folder).contains(where: { $0.hasPrefix("runners/swift/Sources/") && $0.hasSuffix(".swift") })
        else { throw ManifestError.invalid("Swift runner needs Package.swift and reviewable Sources") }
        let receiptURL = try SafePath.resolve("runners/swift/build-receipt.json", inside: folder)
        guard let data = try? Data(contentsOf: receiptURL), let receipt = try? WireJSON.decode(data),
            let hashes = receipt["sha256"]?.object, receipt["command"]?.string?.isEmpty == false,
            receipt["toolchain"]?.string?.isEmpty == false
        else { throw ManifestError.invalid("Swift runner needs a build receipt with command, toolchain and sha256 inventory") }
        let files = try SafePath.files(in: folder).filter {
            $0.hasPrefix("runners/swift/") && $0 != "runners/swift/build-receipt.json" && !$0.hasSuffix("build.sh")
        }
        guard Set(hashes.keys) == Set(files) else { throw ManifestError.invalid("Swift build receipt must cover every source, binary, lock and resource") }
        for path in files {
            guard hashes[path]?.string == (try FileDigest.sha256(SafePath.resolve(path, inside: folder))) else {
                throw ManifestError.invalid("Swift build receipt is stale for \(path)")
            }
        }
    }
}
