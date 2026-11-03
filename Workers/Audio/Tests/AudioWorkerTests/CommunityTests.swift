// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Testing
@testable import AudioWorker

extension PrecisionTests {
    @Test(.heavy) func communityOriginalContainersAndStoredTiers() throws {
        guard let metallib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] else {
            throw audioFailure("Pinned metallib required.")
        }
        GPU.metallib = URL(fileURLWithPath: metallib)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root); Memory.clearCache() }
        let source = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let graph = C1Network(); try NativeParameters.initializeRandom(graph)
        let tensors = Dictionary(uniqueKeysWithValues: graph.parameters().flattened())
        // 234 checkpoint tensors (SincNet window/time buffers are recomputed) plus six PLDA arrays.
        #expect(c1Descriptors.count == 234)
        #expect(tensors.count == c1Descriptors.count + Community1Conversion.pldaArrays.count)
        #expect(Set(c1Descriptors.map(\.name)).union(Community1Conversion.pldaArrays.map { $0.0 }) == Set(tensors.keys))
        for component in ["segmentation", "embedding"] {
            var storage: [String: Data] = [:]
            for d in c1Descriptors where d.component == component {
                var value = tensors[d.name]!
                if let axes = d.permutation {
                    var inverse = axes
                    for (i, axis) in axes.enumerated() { inverse[axis] = i }
                    value = value.transposed(axes: inverse)
                }
                let floats = value.asType(.float32).asArray(Float.self), data = floats.withUnsafeBytes { Data($0) }
                var bytes = storage[d.entry] ?? Data()
                let end = d.offset * 4 + data.count
                if bytes.count < end { bytes.append(Data(count: end - bytes.count)) }
                bytes.replaceSubrange(d.offset * 4..<end, with: data); storage[d.entry] = bytes
            }
            let folder = source.appendingPathComponent(component)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try c1TestZip(storage).write(to: folder.appendingPathComponent("pytorch_model.bin"))
        }
        let pldaFolder = source.appendingPathComponent("plda")
        try FileManager.default.createDirectory(at: pldaFolder, withIntermediateDirectories: true)
        var containers: [String: [String: Data]] = [:]
        for (name, container, array) in Community1Conversion.pldaArrays {
            // The pinned files store mean1, mu, tr and psi as float64 and mean2, lda as float32.
            containers[container, default: [:]][array + ".npy"] = c1TestNumpy(tensors[name]!, double: ["mean1", "mu", "tr", "psi"].contains(array))
        }
        for (container, entries) in containers { try c1TestZip(entries).write(to: pldaFolder.appendingPathComponent(container)) }
        try Data("params: {clustering: {threshold: 0.6, Fa: 0.07, Fb: 0.8}}\n".utf8).write(to: source.appendingPathComponent("config.yaml"))
        let registration = AudioWorker.registrations.first { $0.modelTypes.contains("pyannote_community_1") }!
        let eligible = tensors.filter { c1Eligible($0.key, $0.value.shape) }.count
        #expect(eligible == 18)
        for precision in ["fp16", "int8", "int4"] {
            let output = root.appendingPathComponent(precision)
            let converter = precision == "fp16" ? Community1Conversion.base : Community1Conversion.tiers
            _ = try converter.run(
                [
                    "src_dir": .string((precision == "fp16" ? source : root.appendingPathComponent("fp16")).path), "dst_dir": .string(output.path),
                    "dtype": .string(precision), "group_size": 64, "recipe": .string(converter.recipe)
                ], cancellation: NativeCancellation())
            let adapter = registration.make()
            try adapter.load(
                NativeLoadContext(["model_dir": .string(root.path), "weights_dir": .string(output.path), "dtype": .string(precision), "config": [:]]),
                cancellation: NativeCancellation())
            try NativeParameters.requireHalf(adapter.parameters)
            #expect(adapter.parameters.filter { $0.value.dtype == .uint32 }.count == (precision == "fp16" ? 0 : eligible))
            #expect(adapter.parameters.filter { $0.key.hasPrefix("plda.") }.allSatisfy { $0.value.dtype == .float16 })
            if precision == "fp16" {
                #expect(Set(adapter.parameters.keys) == Set(tensors.keys))
                for (name, expected) in tensors { #expect((adapter.parameters[name]! - expected).abs().max().item(Float.self) == 0, "\(name)") }
            }
            adapter.unload()
        }
        // A tier never derives from the raw containers.
        #expect(throws: (any Error).self) {
            try Community1Conversion.tiers.run(
                [
                    "src_dir": .string(source.path), "dst_dir": .string(root.appendingPathComponent("bad").path), "dtype": "int8", "group_size": 64,
                    "recipe": .string(Community1Conversion.tiers.recipe)
                ], cancellation: NativeCancellation())
        }
    }

    /// References captured once with SciPy 1.17.1 / scikit-learn 1.9.0 / NumPy 2.4.3 and pyannote.audio 4.0.0's vbx.py
    /// (lab/c-audio/mirror/fixture.py) on the same generated inputs.
    @Test func communityClusteringMatchesReferenceLibraries() throws {
        var random = Community1RandomState(seed: 42)
        #expect([random.randomSample(), random.randomSample(), random.randomSample()] == [0.3745401188473625, 0.9507143064099162, 0.7319939418114051])
        let points = (0..<24).map { i -> [Double] in
            let row = (0..<8).map { j in sin(Double(i * 31 + j * 7) * 0.37) * 0.35 + (j == i % 4 ? 1.5 : 0) }
            let norm = sqrt(row.reduce(0) { $0 + $1 * $1 })
            return row.map { $0 / norm }
        }
        let cancellation = NativeCancellation()
        #expect(try Community1Clustering.ahcLabels(points, threshold: 0.6, cancellation: cancellation) == [0, 3, 7, 4, 1, 3, 7, 5, 1, 2, 6, 5, 0, 3, 7, 4, 1, 3, 7, 5, 1, 2, 6, 5])
        #expect(try Community1KMeans.fitPredict(points, clusters: 2, cancellation: cancellation) == [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
        #expect(try Community1KMeans.fitPredict(points, clusters: 3, cancellation: cancellation) == [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1])
        #expect(try Community1KMeans.fitPredict(points, clusters: 5, cancellation: cancellation) == [0, 3, 2, 1, 0, 4, 2, 1, 0, 4, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 4, 2, 1])
        let features: [[Double]] = (0..<24).map { i -> [Double] in
            (0..<6).map { j -> Double in
                let wave: Double = sin(Double(i) * 0.9 + Double(j) * 0.4) * 2
                let bump: Double = (i / 6) == (j % 4) ? 3 : 0
                return wave + bump
            }
        }
        let result = try Community1Clustering.vbx(
            initialLabels: (0..<24).map { $0 / 6 }, features: features, phi: [4, 3, 2, 1.5, 1, 0.5], config: .default, cancellation: cancellation)
        let q0 = [0.19808297636338254, 0.3174300300090338, 0.2725415540110946, 0.21194543961648907]
        let q13 = [0.20483550312539248, 0.3079347384636348, 0.27000854288834353, 0.21722121552262919]
        let sp = [0.2036426684409746, 0.30956310908302515, 0.27048443278956463, 0.21630978968643558]
        for k in 0..<4 {
            #expect(abs(result.responsibilities[0][k] - q0[k]) < 1e-9)
            #expect(abs(result.responsibilities[13][k] - q13[k]) < 1e-9)
            #expect(abs(result.priors[k] - sp[k]) < 1e-9)
        }
        // linear_sum_assignment(maximize=True) on 3 × K, including fewer clusters than speakers.
        #expect(Community1Clustering.assign([[0.9, 0.2, 0.1], [0.8, 0.7, 0.0], [0.1, 0.1, 0.05]]) == [0, 1, 2])
        #expect(Community1Clustering.assign([[0.9, 0.2], [0.95, 0.1], [0.1, 0.3]]) == [-2, 0, 1])
        let cancelled = NativeCancellation(); cancelled.cancel()
        #expect(throws: (any Error).self) {
            try Community1Clustering.vbx(initialLabels: [0, 1], features: [[0, 0], [1, 1]], phi: [1, 1], config: .default, cancellation: cancelled)
        }
    }

    @Test(.heavy) func communityCountReconstructionAndLabels() throws {
        // Two 1 s chunks 0.1 s apart (lab geometry): 56 frames each.
        let timeline = Community1Timeline(chunks: 2, chunkDuration: 1, chunkStep: 0.1)
        #expect(timeline.startFrame(1) == 6)
        #expect(timeline.frames == 66)
        var first = [[Float]](repeating: [0, 0, 0], count: 56), second = first
        for f in 0..<30 { first[f][0] = 1 }
        for f in 20..<56 { first[f][1] = 1 }
        for f in 14..<56 { second[f][0] = 1 }
        let segmentation = [first, second]
        let count = timeline.speakerCount(segmentation)
        // Frame 25: two speakers in chunk 0, one in chunk 1 → mean 1.5 → rint (half to even) 2. Frames 62...65 are uncovered.
        #expect(count[0] == 1 && count[25] == 2 && count[61] == 1 && count[65] == 0)
        let hard = [[0, 1, -2], [1, -2, -2]]
        let regular = timeline.annotate(timeline.reconstruct(segmentation, hard: hard, count: count))
        let exclusive = timeline.annotate(timeline.reconstruct(segmentation, hard: hard, count: count.map { min($0, 1) }))
        #expect(Set(regular.map { $0.cluster }) == [0, 1])
        #expect(regular.contains { a in regular.contains { b in a.cluster != b.cluster && a.start < b.end && b.start < a.end } })
        for a in exclusive { for b in exclusive where a.cluster != b.cluster { #expect(a.end <= b.start || b.end <= a.start) } }
        // Labels sort as strings, so cluster 10 is named before cluster 2.
        #expect(Community1Adapter.labelOrder([2, 10, 0, 1]) == [0, 1, 10, 2])
        let output = Community1Adapter.result(
            [(0.0004, 1.25, "SPEAKER_00")], [(0.0004, 1.25, "SPEAKER_00")], speakers: ["SPEAKER_00"], embeddings: [[Double](repeating: .nan, count: 256)], uri: "rec")
        #expect(output["rttm"]?.string == "SPEAKER rec 1 0.000 1.250 <NA> <NA> SPEAKER_00 <NA> <NA>\n")
        #expect(output["segments"]?.array?.first?["start"]?.number == 0)
        #expect(output["speaker_embeddings"]?.array?.first == .null)
        #expect(Community1Adapter.uri("/tmp/my meeting.wav") == "my_meeting")
        #expect(throws: (any Error).self) { try Community1SpeakerBounds(num: nil, min: 3, max: 2) }
        let bounds = try Community1SpeakerBounds(num: nil, min: 2, max: 2)
        #expect(bounds.exact == 2 && bounds.minimum == 2 && bounds.maximum == 2)
        let open = try Community1SpeakerBounds(num: nil, min: nil, max: nil)
        #expect(open.exact == nil && open.minimum == 1 && open.maximum == nil)
    }
}

/// Frame centres follow pyannote's SlidingWindow[i].middle = 0.5 · (s + (s + duration)), s = i · step, bit for bit
/// (2 Nov). In PyanNet geometry 2,911 of 35,556 ten-minute frames differ in the last bit from i · step + duration / 2;
/// JSON times and RTTM starts do not change, RTTM durations move 1 ms in about 0.7% of segments.
@Test func community1FrameCentresFollowPyannote() {
    let timeline = Community1Timeline(chunks: 1, chunkDuration: 10, chunkStep: 1)
    var binary = [[Bool]](repeating: [false], count: 12)
    for t in 7..<10 { binary[t][0] = true }
    let segments = timeline.annotate(binary)
    #expect(segments.count == 1)
    let s7 = 7 * Community1Config.frameStep, s10 = 10 * Community1Config.frameStep
    #expect(segments.first?.start == 0.5 * (s7 + (s7 + Community1Config.frameDuration)))
    #expect(segments.first?.start == 0.14909375000000002)
    #expect(segments.first?.start != 7 * Community1Config.frameStep + 0.5 * Community1Config.frameDuration)
    #expect(segments.first?.end == 0.5 * (s10 + (s10 + Community1Config.frameDuration)))
}

private func c1TestNumpy(_ array: MLXArray, double: Bool) -> Data {
    let dims = array.shape.map(String.init).joined(separator: ", ") + (array.ndim == 1 ? "," : "")
    var header = "{'descr': '<f\(double ? 8 : 4)', 'fortran_order': False, 'shape': (\(dims)), }"
    while (10 + header.utf8.count + 1) % 64 != 0 { header += " " }; header += "\n"
    var data = Data([147, 78, 85, 77, 80, 89, 1, 0]); data.appendLE(UInt16(header.utf8.count)); data.append(Data(header.utf8))
    if double { data.append(array.asArray(Float.self).map(Double.init).withUnsafeBytes { Data($0) }) } else { data.append(array.asArray(Float.self).withUnsafeBytes { Data($0) }) }
    return data
}
private func c1TestZip(_ entries: [String: Data]) -> Data {
    var data = Data(), central = Data()
    for name in entries.keys.sorted() {
        let bytes = entries[name]!, filename = Data(name.utf8), offset = UInt32(data.count)
        // CRC is not needed by the pinned reader: host verifies the whole source SHA-256.
        data.appendLE(UInt32(0x04034b50)); for v: UInt16 in [20, 0, 0, 0, 0] { data.appendLE(v) }
        for v: UInt32 in [0, UInt32(bytes.count), UInt32(bytes.count)] { data.appendLE(v) }
        data.appendLE(UInt16(filename.count)); data.appendLE(UInt16(0)); data.append(filename); data.append(bytes)
        central.appendLE(UInt32(0x02014b50)); for v: UInt16 in [20, 20, 0, 0, 0, 0] { central.appendLE(v) }
        for v: UInt32 in [0, UInt32(bytes.count), UInt32(bytes.count)] { central.appendLE(v) }
        for v: UInt16 in [UInt16(filename.count), 0, 0, 0, 0] { central.appendLE(v) }
        central.appendLE(UInt32(0)); central.appendLE(offset); central.append(filename)
    }
    let offset = UInt32(data.count); data.append(central); data.appendLE(UInt32(0x06054b50))
    for v: UInt16 in [0, 0, UInt16(entries.count), UInt16(entries.count)] { data.appendLE(v) }
    data.appendLE(UInt32(central.count)); data.appendLE(offset); data.appendLE(UInt16(0)); return data
}
private extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) { var value = value.littleEndian; Swift.withUnsafeBytes(of: &value) { append(contentsOf: $0) } }
}
