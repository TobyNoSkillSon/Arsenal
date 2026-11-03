// SPDX-License-Identifier: MIT
import Foundation

extension CatalogDocument {
    /// The signed inventory is complete: no missing, unlisted or symbolic-link kind files are accepted.
    public func verifyKinds(directory: URL) throws {
        let expected = kindFiles ?? []
        guard !expected.isEmpty else { return } // Legacy v1 catalogs have no shared-schema inventory.
        let present = try SafePath.files(in: directory).filter { URL(fileURLWithPath: $0).lastPathComponent != ".DS_Store" }
        guard Set(present) == Set(expected.map(\.path)) else { throw ManifestError.invalid("unlisted or missing kind files") }
        for file in expected { try FileDigest.verify(SafePath.resolve(file.path, inside: directory), sha256: file.sha256, bytes: file.bytes) }
    }
}
