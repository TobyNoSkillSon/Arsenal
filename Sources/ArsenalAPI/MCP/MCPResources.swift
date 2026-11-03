// SPDX-License-Identifier: MIT
import Foundation
import ImageIO
import UniformTypeIdentifiers
import ArsenalCore
import MCP

extension APIRouter {
    func mcpResources() async -> [Resource] {
        var resources: [Resource] = IntegrationReferences.documents.keys.sorted().map { .init(name: $0, uri: "arsenal://integration/" + $0, mimeType: "text/markdown") }
        for entry in await service.catalog(query: .init()) {
            resources.append(.init(name: entry.id + " card", uri: "arsenal://models/\(entry.id)", mimeType: "application/json"))
            resources.append(.init(name: entry.id + " README", uri: "arsenal://models/\(entry.id)/README", mimeType: "text/markdown"))
            for operation in await service.operations(id: entry.id) {
                resources.append(
                    .init(
                        name: entry.id + " " + operation.name + " schema", uri: "arsenal://models/\(entry.id)/operations/\(operation.name)/schema",
                        mimeType: "application/schema+json"))
                if operation.outputSchema != nil {
                    resources.append(
                        .init(
                            name: entry.id + " " + operation.name + " output schema", uri: "arsenal://models/\(entry.id)/operations/\(operation.name)/output-schema",
                            mimeType: "application/schema+json"))
                }
            }
        }
        for id in mcpJobs.keys.sorted() { resources.append(.init(name: "Job " + id, uri: "arsenal://jobs/" + id, mimeType: "application/json")) }
        return resources
    }
    func readMCPResource(_ uri: String) async throws -> Resource.Content {
        if uri.hasPrefix("arsenal://integration/"), let document = IntegrationReferences.documents[String(uri.dropFirst("arsenal://integration/".count))] {
            return .text(document, uri: uri, mimeType: "text/markdown")
        }
        if uri.hasPrefix("arsenal://models/"), uri.hasSuffix("/README") {
            let id = String(uri.dropFirst("arsenal://models/".count).dropLast("/README".count))
            return .text(try await service.modelREADME(id: id), uri: uri, mimeType: "text/markdown")
        }
        if uri.hasPrefix("arsenal://models/"), !uri.dropFirst("arsenal://models/".count).contains("/") {
            let id = String(uri.dropFirst("arsenal://models/".count))
            guard let entry = await mcpCatalog().first(where: { $0.id == id }) else { throw MCPError.invalidParams("Unknown model; call \(MCPNames.prefix)find.") }
            let object = try WireJSON.decode(JSONEncoder().encode(entry))
            return .text(String(decoding: try WireJSON.encode(object), as: UTF8.self), uri: uri, mimeType: "application/json")
        }
        if uri.hasPrefix("arsenal://outputs/") {
            let id = String(uri.dropFirst("arsenal://outputs/".count))
            guard let url = mcpOutputs[id], let bytes = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize, bytes <= 64 * 1024 * 1024 else {
                throw MCPError.invalidParams("Output not found; use the path printed in the result.")
            }
            return .binary(try Data(contentsOf: outputPath(url.path)), uri: uri, mimeType: contentType(url))
        }
        if uri.hasPrefix("arsenal://jobs/"), let job = mcpJobs[String(uri.dropFirst("arsenal://jobs/".count))] {
            return .text(String(decoding: try WireJSON.encode(await mcpJobObject(job)), as: UTF8.self), uri: uri, mimeType: "application/json")
        }
        let parts = uri.dropFirst("arsenal://models/".count).split(separator: "/").map(String.init)
        guard uri.hasPrefix("arsenal://models/"), parts.count == 4, parts[1] == "operations", ["schema", "output-schema"].contains(parts[3]),
            let operation = await service.operations(id: parts[0]).first(where: { $0.name == parts[2] })
        else {
            throw MCPError.invalidParams("Resource not found; \(MCPNames.prefix)model(model: ID) prints each operation's schema URI.")
        }
        return .text(
            String(decoding: try WireJSON.encode(parts[3] == "output-schema" ? operation.outputSchema ?? [:] : operation.inputSchema), as: UTF8.self), uri: uri,
            mimeType: "application/schema+json")
    }
}
func mcpThumbnail(_ url: URL) -> Data? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let image = CGImageSourceCreateThumbnailAtIndex(
            source, 0,
            [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: 256,
                kCGImageSourceCreateThumbnailWithTransform: true
            ] as CFDictionary)
    else { return nil }
    let data = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { return nil }
    return data as Data
}
