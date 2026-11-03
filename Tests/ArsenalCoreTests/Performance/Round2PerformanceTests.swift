// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import ArsenalCore

private actor TimedCatalogSource: CatalogSource {
    let base: BundleCatalogSource
    var readMS: Double = 0
    var reads = 0
    var batches = 0
    init(_ base: BundleCatalogSource) { self.base = base }
    func read(path: String, commit: String) async throws -> Data {
        let start = ContinuousClock.now
        defer { let elapsed = ContinuousClock.now - start; readMS += Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15; reads += 1 }
        return try await base.read(path: path, commit: commit)
    }
    func readAvailable(paths: [String], commit: String) async -> [String: Data] {
        let start = ContinuousClock.now
        let result = await base.readAvailable(paths: paths, commit: commit)
        let elapsed = ContinuousClock.now - start
        readMS += Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
        reads += paths.count; batches += 1
        return result
    }
}

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_CORE2_PROBE"] == "1"))
struct Round2PerformanceTests {
    @Test func getProgressEventRate() async throws {
        let payload = Data(repeating: 42, count: 16 * 1024 * 1024)
        let fixture = try await makeLocalCatalogFixture(kinds: ["voices"], filePayloads: ["base.bin": payload])
        defer { fixture.stop() }
        let events = await fixture.service.events()
        let started = ContinuousClock.now
        let get = Task { try await fixture.service.get(id: "fixture-voices", precision: "bf16") }
        var count = 0
        var first: ContinuousClock.Instant?, last: ContinuousClock.Instant?
        for await event in events {
            if case .progress = event { count += 1; if first == nil { first = .now }; last = .now }
            if case .rowsChanged = event, await fixture.service.rows().first?.states["bf16"]?.state == .ready { break }
        }
        try await get.value
        let elapsed = ContinuousClock.now - started
        let span = first.flatMap { first in last.map { $0 - first } } ?? .zero
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let spanSeconds = Double(span.components.seconds) + Double(span.components.attoseconds) / 1e18
        print("PERF core2 get-bytes=\(payload.count) progress-events=\(count) elapsed-s=\(seconds) progress-span-s=\(spanSeconds)")
        #expect(await fixture.service.rows().first?.states["bf16"]?.progress == 1)
        if ProcessInfo.processInfo.environment["ARSENAL_EXPECT_THROTTLE"] == "1" { #expect(Double(count) <= ceil(spanSeconds * 10) + 1) }
        await fixture.service.shutdown()
    }
    @Test func bundledLaunchAndRefresh() async throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = URL(fileURLWithPath: "/tmp/arsenal-mb-pol-core2/probe-" + UUID().uuidString)
        let resources = root.appendingPathComponent("Resources")
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        for name in ["models", "kinds"] { try FileManager.default.copyItem(at: repo.appendingPathComponent(name), to: resources.appendingPathComponent(name)) }
        func inventory(_ dir: URL) throws -> [CatalogFile] {
            try SafePath.files(in: dir).map {
                CatalogFile(path: $0, sha256: try FileDigest.sha256(dir.appendingPathComponent($0)), bytes: try FileDigest.bytes(dir.appendingPathComponent($0)))
            }
        }
        let kinds = resources.appendingPathComponent("kinds")
        let entries = try FileManager.default.contentsOfDirectory(at: resources.appendingPathComponent("models"), includingPropertiesForKeys: nil).sorted { $0.path < $1.path }
            .compactMap { folder -> CatalogModel? in
                guard let manifest = try? ModelManifest.load(folder, kindsDirectory: kinds) else { return nil }
                return CatalogModel(manifest: manifest, files: try inventory(folder))
            }
        let data = try CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: entries, kindFiles: try inventory(kinds)).encoded()
        let key = Curve25519.Signing.PrivateKey()
        try FileManager.default.createDirectory(at: resources.appendingPathComponent("catalog"), withIntermediateDirectories: true)
        try data.write(to: resources.appendingPathComponent("catalog/catalog.json"))
        try key.signature(for: data).write(to: resources.appendingPathComponent("catalog/catalog.json.sig"))
        try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
        let clock = ContinuousClock()
        func ms(_ duration: Duration) -> Double { Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15 }
        for sample in 1...3 {
            let paths = ArsenalPaths(support: root.appendingPathComponent("support-\(sample)"), cache: root.appendingPathComponent("cache-\(sample)"))
            let started = clock.now
            let source = try CatalogBootstrap.verifiedBundleSource(resources: resources)
            let sourceMS = ms(clock.now - started)
            let selectStart = clock.now
            _ = try CatalogBootstrap.select(
                bundledDirectory: resources.appendingPathComponent("catalog"), cachedDirectory: paths.directory("catalog"), publicKey: key.publicKey.rawRepresentation)
            let selectMS = ms(clock.now - selectStart)
            let openStart = clock.now
            let service = try await RealModelsService.open(repository: resources, resources: resources, paths: paths)
            let openMS = ms(clock.now - openStart)
            for entry in entries { try await service.add(id: entry.id) }
            _ = await service.rows(); _ = await service.catalog()
            var tick = clock.now
            for _ in 0..<30 { _ = await service.rows() }
            let rowsMS = ms(clock.now - tick) / 30
            tick = clock.now
            for _ in 0..<30 { _ = await service.catalog() }
            let catalogMS = ms(clock.now - tick) / 30
            print(
                "PERF core2 sample=\(sample) models=\(entries.count) bundle-source-ms=\(sourceMS) select-ms=\(selectMS) open-ms=\(openMS) rows-ms=\(rowsMS) catalog-ms=\(catalogMS) source=\(source.catalog.models.count)"
            )
            await service.shutdown()
            if sample == 1 {
                let measuredSource = TimedCatalogSource(source)
                let tracedPaths = ArsenalPaths(support: root.appendingPathComponent("traced-support"))
                let traceStart = clock.now
                let traced = try await RealModelsService.open(
                    catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, source: measuredSource, kinds: [], paths: tracedPaths,
                    kindsDirectory: kinds)
                print(
                    "PERF core2 traced-open-ms=\(ms(clock.now - traceStart)) source-reads=\(await measuredSource.reads) source-batches=\(await measuredSource.batches) source-read-ms=\(await measuredSource.readMS)"
                )
                await traced.shutdown()
            }
        }
    }
}
