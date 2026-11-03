// SPDX-License-Identifier: MIT
// ESM3 open (biohub/esm3-sm-open-v1) as a kit runner: the 1.4B multi-track transformer, the structure tokenizer
// (VQ-VAE encoder and decoder) and the function-token decoder, ported to MLX from the maker's esm package (commit
// 8f0834c). Get converts the pinned PyTorch archives and NumPy arrays to safetensors (kit Checkpoints.swift); nothing
// here reads a pickle.
import Foundation
import MLX
import MLXNN

let modelKind = "local.protein"
let modelOps = ["embed_sequence", "predict_masked", "score_variants", "generate", "fold", "inverse_fold", "predict_function", "encode_structure"]

final class Model {
    let net: ESM3Net
    let encoder: StructureEncoder
    let decoder: StructureDecoder
    let functionDecoder: FunctionDecoder
    let functionTokenizer: FunctionTokenizer
    let interproIDs: [String]
    let interproNames: [String: String]
    let keywords: [String]
    let residueVocabulary: [String]
    /// The runner's cap on residues per request (memory of the pairwise heads grows with the square).
    static let maxResidues = 1024
    /// Lab only: when set, requests save their intermediate tensors here for the parity check (never set by Arsenal).
    let captureDir = ProcessInfo.processInfo.environment["ARSENAL_ESM3_CAPTURE_DIR"].map { URL(fileURLWithPath: $0) }

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        func file(_ name: String) throws -> URL {
            let url = weightsDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw RunnerError.failed("\(name) is missing from the weights; Get converts the pinned .pth and .npy/.npz files to safetensors.")
            }
            return url
        }
        func tensors(_ name: String, skip: (String) -> Bool = { _ in false }) throws -> [String: MLXArray] {
            var out: [String: MLXArray] = [:]
            for (key, value) in try loadArrays(url: try file(name)) where !skip(key) {
                out[CheckpointKeys.rename(key)] = value.dtype == .bfloat16 ? value.asType(.float16) : value
            }
            return out
        }
        net = ESM3Net()
        try Weights.apply(try tensors("data/weights/esm3_sm_open_v1.safetensors"), to: net)
        encoder = StructureEncoder()
        // The codebook's EMA statistics (N, z_avg) are training state.
        try Weights.apply(try tensors("data/weights/esm3_structure_encoder_v0.safetensors") { $0 == "codebook.N" || $0 == "codebook.z_avg" }, to: encoder)
        decoder = StructureDecoder()
        try Weights.apply(try tensors("data/weights/esm3_structure_decoder_v0.safetensors"), to: decoder)
        functionDecoder = FunctionDecoder()
        // The keyword TF-IDF regression head is not used by decoding (FunctionTokenDecoder.decode).
        try Weights.apply(try tensors("data/weights/esm3_function_decoder_v0.safetensors") { $0.hasPrefix("heads.keyword_tfidf.") }, to: functionDecoder)

        keywords = try TokenData.lines(try file("data/keyword_vocabulary_safety_filtered_58641.txt"))
        guard let idf = try HostTensors.read(try file("data/keyword_idf_safety_filtered_58641.safetensors"))["array"]?.values else {
            throw RunnerError.failed("The keyword IDF array is missing.")
        }
        let planes = try HostTensors.read(try file("data/hyperplanes_8bit_58641.safetensors"))
        let hyperplanes = try (0 ..< 8).map { (i: Int) -> [Double] in
            guard let table = planes[String(i)] else { throw RunnerError.failed("LSH table \(i) is missing.") }
            return table.values
        }
        let residue = try TokenData.residueLabels(try file("data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv"))
        residueVocabulary = ["<pad>", "<motif>", "<unk>", "<none>"] + residue.labels
        guard residueVocabulary.count == ESM3Vocab.residueVocab else {
            throw RunnerError.failed("The residue annotation vocabulary has \(residueVocabulary.count) entries; the model has \(ESM3Vocab.residueVocab).")
        }
        do {
            functionTokenizer = try FunctionTokenizer(
                tfidf: try TFIDF(vocabulary: keywords, idf: idf),
                interproKeywords: try TokenData.interproKeywords(try file("data/interpro_29026_to_keywords_58641.csv")),
                hyperplanes: hyperplanes, residueLabels: residue.labels)
            (interproIDs, interproNames) = try TokenData.interproEntries(try file("data/entry_list_safety_29026.list"))
        } catch let error as TokenError { throw RunnerError.failed(error.description) }
        guard interproIDs.count == 29026, keywords.count == 58641 else { throw RunnerError.failed("The function vocabularies do not match the function decoder.") }
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        do {
            switch op {
            case "embed_sequence": return try embed(params)
            case "predict_masked": return try predictMasked(params)
            case "score_variants": return try scoreVariants(params)
            case "generate": return try generate(params)
            case "fold": return try fold(params)
            case "inverse_fold": return try inverseFold(params)
            case "predict_function": return try predictFunction(params)
            case "encode_structure": return try encodeStructure(params)
            default: throw RunnerError.invalid("unknown op \(op)")
            }
        } catch let error as TokenError {
            throw RunnerError.invalid(error.description)
        }
    }

    // MARK: prompts

    /// A protein prompt across tracks, from the request (sequence, length, structure file, secondary structure, SASA,
    /// function annotations), as ESM3.encode builds it.
    struct Prompt {
        var tracks: Tracks
        var length: Int
        /// The structure file's chain, when one was given (its sequence and the residues placed in the prompt).
        var source: Backbone?
    }

    func prompt(_ params: Params, sequenceRequired: Bool = false, allowMasks: Bool = true) throws -> Prompt {
        let raw = params.raw
        var backbone: Backbone?
        if raw["structure"] != nil {
            let url = try params.file("structure")
            backbone = try PDB.parse(try PDB.read(url), chain: (raw["chain"] as? String).flatMap { $0.isEmpty ? nil : $0 }, maxResidues: 10 * Self.maxResidues)
        }
        // The structure segment used: residues from..to of the file's chain (1-indexed), placed at prompt position `at`.
        var segment: ClosedRange<Int>?
        if let backbone {
            let range = raw["structure_residues"] as? [Any]
            let from = (range?.first as? NSNumber)?.intValue ?? 1, to = (range?.last as? NSNumber)?.intValue ?? backbone.count
            guard 1 <= from, from <= to, to <= backbone.count else { throw RunnerError.invalid("structure_residues must lie within 1-\(backbone.count).") }
            segment = from ... to
            try backbone.requireBackbone(segment)
        }
        var sequenceIDs: [Int]?
        if let text = raw["sequence"] as? String {
            sequenceIDs = try SequenceTokens.encode(text)
            if !allowMasks, sequenceIDs!.contains(ESM3Vocab.sequenceMask) { throw RunnerError.invalid("sequence may not contain _ masks here.") }
        } else if sequenceRequired {
            throw RunnerError.invalid("sequence is required.")
        }
        let explicitLength = (raw["length"] as? NSNumber)?.intValue
        let length: Int
        if let sequenceIDs { length = sequenceIDs.count - 2 } else if let explicitLength { length = explicitLength } else if let segment { length = segment.count } else {
            throw RunnerError.invalid("Give a sequence, a length or a structure.")
        }
        guard length >= 1, length <= Self.maxResidues else { throw RunnerError.invalid("The protein must have 1 to \(Self.maxResidues) residues; this one has \(length).") }
        if let explicitLength, explicitLength != length { throw RunnerError.invalid("length is \(explicitLength) but the sequence has \(length) residues.") }
        var t = Tracks.empty(length: length)
        if let sequenceIDs { t.sequence = sequenceIDs }
        if let ss = raw["secondary_structure"] as? String { t.ss8 = try SS8Tokens.encode(ss, length: length) }
        if let values = raw["sasa"] as? [Any] {
            t.sasa = try SASATokens.encode(values.map { ($0 as? NSNumber)?.doubleValue }, length: length)
        }
        if let list = raw["function"] as? [Any] {
            let annotations = try list.map { item -> Annotation in
                guard let o = item as? [String: Any], let label = o["label"] as? String, let start = (o["start"] as? NSNumber)?.intValue,
                    let end = (o["end"] as? NSNumber)?.intValue
                else { throw RunnerError.invalid("Each function annotation needs label, start and end.") }
                return Annotation(label: label, start: start, end: end)
            }
            t.function = try functionTokenizer.encode(annotations, length: length)
        }
        if let backbone, let segment {
            let at = (raw["structure_at"] as? NSNumber)?.intValue ?? 1
            guard at >= 1, at + segment.count - 1 <= length else {
                throw RunnerError.invalid("The structure segment (\(segment.count) residues at position \(at)) does not fit a protein of \(length) residues.")
            }
            let atoms = Array(backbone.atoms[(segment.lowerBound - 1) ..< segment.upperBound])
            let (tokens, latent) = encoder.encode(atoms)
            save("encoder", ["latent": latent])
            for (k, row) in atoms.enumerated() {
                t.coordinates[at + k] = row
                t.structure[at + k] = tokens[k]
            }
            // ESMProtein.from_protein_chain also gives the file's sequence; it fills the residues the request left open.
            if sequenceIDs == nil {
                for (k, letter) in backbone.sequence[(segment.lowerBound - 1) ..< segment.upperBound].enumerated() {
                    if params.bool("structure_sequence", default: true) { t.sequence[at + k] = ESM3Vocab.sequenceIndex[String(letter)] ?? ESM3Vocab.sequenceIndex["X"]! }
                }
            }
        }
        return Prompt(tracks: t, length: length, source: backbone)
    }

    // MARK: forward and outputs

    func forward(_ t: Tracks, params: Params, capture: String? = nil) throws -> Logits {
        var taps: [String: MLXArray] = [:]
        let tapping = capture != nil && captureDir != nil
        let logits = try net(t, taps: tapping ? [0, 1, 23, 47] : [], tap: tapping ? { taps[$0] = $1[0].asType(.float32) } : nil, cancelled: { try params.checkCancelled() })
        eval(logits.sequence, logits.structure, logits.ss8, logits.sasa, logits.function, logits.residue, logits.embeddings)
        if let capture, tapping {
            taps["embeddings"] = logits.embeddings
            for (name, value) in [("sequence", logits.sequence), ("structure", logits.structure), ("secondary_structure", logits.ss8), ("sasa", logits.sasa),
                                  ("function", logits.function), ("residue_annotations", logits.residue)] { taps["logits." + name] = value }
            save(capture, taps)
        }
        return logits
    }

    func save(_ prefix: String, _ tensors: [String: MLXArray]) {
        guard let captureDir, CaptureName.sanitized(prefix) != nil else { return }
        try? FileManager.default.createDirectory(at: captureDir, withIntermediateDirectories: true)
        try? MLX.save(arrays: tensors.mapValues { $0.asType(.float32) }, url: captureDir.appendingPathComponent(prefix + ".safetensors"))
    }

    static func rows(_ a: MLXArray) -> [[Double]] {
        let width = a.dim(-1), flat = a.asType(.float32).asArray(Float.self)
        return stride(from: 0, to: flat.count, by: width).map { flat[$0 ..< ($0 + width)].map(Double.init) }
    }

    /// Function annotations decoded from function tokens [L + 2][8] (FunctionTokenDecoder.decode and
    /// decode_function_tokens): keywords first, then InterPro entries as "name (IPR…)"; positions are residues.
    func decodeFunction(_ rows: [[Int]], interproThreshold: Double = 0.1, keywordThreshold: Double = 0.5, capture: String? = nil) -> [[String: Any]] {
        let decode = rows.map { row in row.allSatisfy { $0 != 0 && $0 != ESM3Vocab.functionNone && $0 != 2 } }
        guard decode.contains(true) else { return [] }
        let (interpro, keyword) = functionDecoder(rows)
        if let capture { save(capture, ["interpro_logits": interpro, "keyword_logits": keyword]) }
        func hits(_ logits: MLXArray, _ threshold: Double) -> [(Int, Int)] {
            let cut = Float(Foundation.log(threshold / (1 - threshold)))
            let flags = (logits .>= MLXArray(cut)).asType(.uint8).asArray(UInt8.self)
            let width = logits.dim(-1)
            var out: [(Int, Int)] = []
            for (i, flag) in flags.enumerated() where flag != 0 && decode[i / width] { out.append((i / width, i % width)) }
            return out
        }
        var keywordAnnotations: [Annotation] = []
        var byKeyword: [Int: [Range<Int>]] = [:], order: [Int] = []
        for (position, id) in hits(keyword, keywordThreshold) {
            if byKeyword[id] == nil { order.append(id) }
            byKeyword[id, default: []].append(position ..< (position + 1))
        }
        for id in order {
            for r in Annotations.mergeRanges(byKeyword[id]!, gap: 0) { keywordAnnotations.append(Annotation(label: keywords[id], start: r.lowerBound, end: r.upperBound - 1)) }
        }
        keywordAnnotations = Annotations.dropShort(Annotations.merge(keywordAnnotations, gap: 3), minLength: 5)
        var interproAnnotations = hits(interpro, interproThreshold).map { Annotation(label: interproIDs[$0.1], start: $0.0, end: $0.0) }
        interproAnnotations = Annotations.dropShort(Annotations.merge(interproAnnotations, gap: 3), minLength: 5)
        return keywordAnnotations.map { ["label": $0.label, "start": $0.start, "end": $0.end, "kind": "keyword"] }
            + interproAnnotations.map { a in
                let name = interproNames[a.label]
                return ["label": name.map { "\($0) (\(a.label))" } ?? a.label, "start": a.start, "end": a.end, "kind": "interpro", "interpro": a.label] as [String: Any]
            }
    }

    /// Residue annotations from the residue head (sample_residue_annotation_logits, decode_residue_annotation_tokens).
    func decodeResidues(_ logits: MLXArray) -> [[String: Any]] {
        let L = logits.dim(0)
        let rows = Self.rows(logits)
        var found: [Annotation] = []
        // The 16 highest logits per position whose sigmoid is at least 0.5, read slot by slot as the maker's decoder.
        let top = (0 ..< L).map { position -> [Int] in
            guard position > 0, position < L - 1 else { return [] }
            let row = rows[position]
            return Array(row.indices.sorted { row[$0] != row[$1] ? row[$0] > row[$1] : $0 < $1 }.prefix(ESM3Vocab.residueSlots))
        }
        for slot in 0 ..< ESM3Vocab.residueSlots {
            for position in 1 ..< (L - 1) {
                let id = top[position][slot]
                guard rows[position][id] >= 0, id >= 4 else { continue }
                found.append(Annotation(label: residueVocabulary[id], start: position, end: position))
            }
        }
        return Annotations.dropShort(Annotations.merge(found, gap: 3), minLength: 5).map { ["label": $0.label, "start": $0.start, "end": $0.end] }
    }

    // MARK: ops

    func embed(_ params: Params) throws -> [String: Any] {
        let p = try prompt(params, sequenceRequired: true)
        if params.bool("per_residue", default: false) { try RequestLimits.reply((p.length + 1) * ESM3Net.dimensions, "per_residue for \(p.length) residues") }
        let logits = try forward(p.tracks, params: params)
        let perResidue = logits.embeddings[1 ..< (p.length + 1)]
        let pooled = perResidue.mean(axis: 0)
        eval(pooled)
        var out: [String: Any] = ["embedding": list(pooled), "dimensions": pooled.dim(0), "residues": p.length]
        if params.bool("per_residue", default: false) { out["per_residue"] = Self.rows(perResidue) }
        return out
    }

    func predictMasked(_ params: Params) throws -> [String: Any] {
        let p = try prompt(params)
        let requested = params.raw["return_logits"] as? [String] ?? []
        try RequestLimits.logits(requested, residues: p.length)
        let logits = try forward(p.tracks, params: params, capture: capture(params))
        if let capture = capture(params) {
            save(capture + ".tokens", [
                "sequence": MLXArray(p.tracks.sequence.map(Int32.init)), "structure": MLXArray(p.tracks.structureInput.map(Int32.init)),
                "secondary_structure": MLXArray(p.tracks.ss8.map(Int32.init)), "sasa": MLXArray(p.tracks.sasa.map(Int32.init)),
                "function": MLXArray(p.tracks.function.flatMap { $0.map(Int32.init) }, [p.length + 2, 8]),
            ])
            _ = decodeFunction(p.tracks.function, capture: capture + ".function_decoder")
        }
        let topK = Swift.max(1, Swift.min(params.int("top_k", default: 5), 20))
        let seqRows = Self.rows(logits.sequence)
        let valid = Set(ESM3Vocab.sequenceValid)
        var filled = p.tracks.sequence
        var masked: [[String: Any]] = []
        for i in 1 ... p.length where p.tracks.sequence[i] == ESM3Vocab.sequenceMask {
            let logp = Sampling.logSoftmax(seqRows[i].indices.map { valid.contains($0) ? seqRows[i][$0] : -.infinity })
            let order = ESM3Vocab.sequenceValid.sorted { logp[$0] != logp[$1] ? logp[$0] > logp[$1] : $0 < $1 }
            filled[i] = order[0]
            masked.append(["position": i, "residue": ESM3Vocab.sequence[order[0]],
                           "top": order.prefix(topK).map { ["residue": ESM3Vocab.sequence[$0], "probability": Foundation.exp(logp[$0])] }])
        }
        let ssRows = Self.rows(logits.ss8), sasaRows = Self.rows(logits.sasa)
        let ss = (1 ... p.length).map { i -> String in
            let best = ESM3Vocab.ss8Valid.max { ssRows[i][$0] < ssRows[i][$1] }!
            return ESM3Vocab.ss8[best]
        }.joined()
        let sasaValid = Set(ESM3Vocab.sasaValid)
        let sasa: [Any] = (1 ... p.length).map { i -> Any in
            let probs = Sampling.logSoftmax(sasaRows[i].indices.map { sasaValid.contains($0) ? sasaRows[i][$0] : -.infinity }).map { Foundation.exp($0) }
            let value = SASATokens.expected(probabilities: probs)
            return value.isFinite ? value : NSNull()
        }
        var out: [String: Any] = [
            "length": p.length, "sequence": SequenceTokens.decode(filled), "masked": masked, "secondary_structure": ss, "sasa": sasa,
            "residue_annotations": decodeResidues(logits.residue),
        ]
        if !requested.isEmpty {
            var tracks: [String: Any] = [:]
            for name in requested {
                let a: MLXArray
                switch name {
                case "sequence": a = logits.sequence
                case "structure": a = logits.structure
                case "secondary_structure": a = logits.ss8
                case "sasa": a = logits.sasa
                case "function": a = logits.function.reshaped(p.length + 2, -1)
                case "residue_annotations": a = logits.residue
                default: throw RunnerError.invalid("return_logits names an unknown track \(name).")
                }
                tracks[name] = Self.rows(a[1 ..< (p.length + 1)])
            }
            out["logits"] = tracks
        }
        return out
    }

    func scoreVariants(_ params: Params) throws -> [String: Any] {
        let p = try prompt(params, sequenceRequired: true, allowMasks: false)
        let method = params.string("method", default: "masked_marginal")
        guard ["masked_marginal", "wildtype_marginal"].contains(method) else { throw RunnerError.invalid("method must be masked_marginal or wildtype_marginal.") }
        let variants = try params.strings("variants")
        guard variants.count <= 10_000 else { throw RunnerError.invalid("At most 10,000 variants per request.") }
        let parsed = try variants.map { try Variants.parse($0, sequence: p.tracks.sequence) }
        var logp: [Int: [Double]] = [:]
        if method == "wildtype_marginal" {
            let rows = Self.rows(try forward(p.tracks, params: params).sequence)
            for position in Set(parsed.flatMap { $0.map(\.position) }) { logp[position] = Sampling.logSoftmax(rows[position]) }
        } else {
            let positions = Set(parsed.flatMap { $0.map(\.position) }).sorted()
            guard positions.count <= 512 else { throw RunnerError.invalid("Masked-marginal scoring takes at most 512 distinct positions per request.") }
            for position in positions {
                var t = p.tracks
                t.sequence[position] = ESM3Vocab.sequenceMask
                logp[position] = Sampling.logSoftmax(Self.rows(try forward(t, params: params).sequence)[position])
            }
        }
        let scores = zip(variants, parsed).map { text, mutations -> [String: Any] in
            ["variant": text, "score": mutations.reduce(0) { $0 + logp[$1.position]![$1.mutant] - logp[$1.position]![$1.wild] }]
        }
        return ["method": method, "scores": scores, "residues": p.length]
    }

    // MARK: iterative generation

    struct GenerationSettings {
        var steps: Int?
        var temperature: Double
        var annealing: Bool
        var topP: Double
        var schedule: String
        var strategy: String
        var seed: UInt64

        /// `steps` defaults to the maker's GenerationConfig (20) for generate, and to residues / 16 (the maker's folding,
        /// inverse folding and function examples) when `defaultSteps` is nil.
        init(_ params: Params, defaults: (temperature: Double, annealing: Bool, strategy: String), defaultSteps: Int? = nil) throws {
            steps = (params.raw["num_steps"] as? NSNumber)?.intValue ?? defaultSteps
            temperature = params.double("temperature", default: defaults.temperature)
            annealing = params.bool("temperature_annealing", default: defaults.annealing)
            topP = params.double("top_p", default: 1)
            schedule = params.string("schedule", default: "cosine")
            strategy = params.string("strategy", default: defaults.strategy)
            seed = (params.raw["seed"] as? NSNumber)?.uint64Value ?? UInt64.random(in: 0 ... UInt64(Int32.max))
            guard Schedule.names.contains(schedule) else { throw RunnerError.invalid("schedule must be cosine or linear.") }
            guard ["random", "entropy"].contains(strategy) else { throw RunnerError.invalid("strategy must be random or entropy.") }
            guard temperature >= 0, temperature <= 10, topP > 0, topP <= 1 else { throw RunnerError.invalid("temperature must be 0-10 and top_p in (0, 1].") }
            if let steps, steps < 1 || steps > 4096 { throw RunnerError.invalid("num_steps must be 1-4096.") }
        }
    }

    static let trackNames = ["sequence", "structure", "secondary_structure", "sasa", "function"]

    /// The passes iterate runs for `track`: num_steps (default residues / 16), capped at the masked positions.
    static func plannedSteps(_ track: String, _ t: Tracks, _ s: GenerationSettings) -> Int {
        let total = (0 ..< t.length).filter { t.isMasked(track, $0) }.count
        return Swift.min(s.steps ?? Swift.max(1, (t.length - 2) / 16), total)
    }

    /// iterative_sampling_tokens for one track of one prompt. Returns the number of steps run.
    @discardableResult
    func iterate(_ track: String, _ t: inout Tracks, _ s: GenerationSettings, rng: inout SeededGenerator, params: Params) throws -> Int {
        let L = t.length
        let total = (0 ..< L).filter { t.isMasked(track, $0) }.count
        guard total > 0 else { throw RunnerError.invalid("Cannot generate \(track): the prompt has no masked positions in that track.") }
        let steps = Self.plannedSteps(track, t, s)
        for step in 0 ..< steps {
            let logits = try forward(t, params: params)
            let temperature = s.annealing ? Schedule.annealed(s.temperature, step: step, steps: steps) : s.temperature
            let candidates = (0 ..< L).filter { t.isMasked(track, $0) }
            var entropy = [Double](repeating: .infinity, count: L)
            var proposal: [Int: Any] = [:]
            func integer(_ a: MLXArray, valid: [Int]) {
                let rows = Self.rows(a), set = Set(valid)
                for i in candidates {
                    let (token, logp) = Sampling.sample(rows[i], valid: set, temperature: temperature, topP: s.topP, rng: &rng)
                    proposal[i] = token
                    entropy[i] = Sampling.entropy(logp)
                }
            }
            switch track {
            case "sequence": integer(logits.sequence, valid: ESM3Vocab.sequenceValid)
            case "structure": integer(logits.structure, valid: ESM3Vocab.structureValid)
            case "secondary_structure": integer(logits.ss8, valid: ESM3Vocab.ss8Valid)
            case "sasa": integer(logits.sasa, valid: ESM3Vocab.sasaValid)
            default:
                let flat = logits.function.asArray(Float.self)
                for i in candidates {
                    let rows = (0 ..< 8).map { d in (0 ..< 260).map { Double(flat[(i * 8 + d) * 260 + $0]) } }
                    let (ids, logp) = Sampling.function(rows, temperature: temperature, topP: s.topP)
                    proposal[i] = ids
                    entropy[i] = logp.reduce(0) { $0 + Sampling.entropy($1) }
                }
            }
            let count = Schedule.toSample(s.schedule, step: step, steps: steps, stillMasked: candidates.count, total: total)
            for i in Sampling.choose(candidates: candidates, count: count, strategy: s.strategy, entropy: entropy, rng: &rng) {
                switch track {
                case "sequence": t.sequence[i] = proposal[i] as! Int
                case "structure": t.structure[i] = proposal[i] as! Int
                case "secondary_structure": t.ss8[i] = proposal[i] as! Int
                case "sasa": t.sasa[i] = proposal[i] as! Int
                default: t.function[i] = proposal[i] as! [Int]
                }
            }
            try params.checkCancelled()
        }
        return steps
    }

    /// The decoded structure of a prompt: from complete structure tokens (decoder), else its own coordinates.
    func structureOutput(_ t: Tracks, capture: String? = nil) -> (atoms: [[Vec3]], plddt: [Double]?, ptm: Double?, pae: MLXArray?)? {
        let L = t.length
        let inner = t.structure[1 ..< (L - 1)]
        if !inner.contains(ESM3Vocab.structureMask) {
            let decoded = decoder.decode(t.structureInput)
            eval(decoded.backbone, decoded.plddt, decoded.pae)
            if let capture { save(capture, ["backbone": decoded.backbone, "plddt": decoded.plddt, "pae": decoded.pae, "ptm": MLXArray([decoded.ptm])]) }
            let flat = decoded.backbone.asArray(Float.self)
            let atoms = (1 ..< (L - 1)).map { i in (0 ..< 3).map { a -> Vec3 in let o = (i * 3 + a) * 3; return (Double(flat[o]), Double(flat[o + 1]), Double(flat[o + 2])) } }
            let plddt = decoded.plddt.asArray(Float.self)[1 ..< (L - 1)].map(Double.init)
            return (atoms, plddt, Double(decoded.ptm), decoded.pae[1 ..< (L - 1), 1 ..< (L - 1)])
        }
        let rows = t.coordinates[1 ..< (L - 1)]
        guard rows.allSatisfy({ $0.allSatisfy { $0 != nil } }) else { return nil }
        return (rows.map { $0.map { $0! } }, nil, nil, nil)
    }

    func describe(_ t: Tracks, generated: Set<String>, params: Params, capture: String? = nil) -> [String: Any] {
        let L = t.length
        var out: [String: Any] = ["length": L - 2, "sequence": SequenceTokens.decode(t.sequence)]
        if t.ss8[1 ..< (L - 1)].contains(where: { $0 != 0 }) { out["secondary_structure"] = SS8Tokens.decode(t.ss8) }
        if t.sasa[1 ..< (L - 1)].contains(where: { $0 != 0 }) { out["sasa"] = SASATokens.decode(t.sasa).map { $0.map { $0 as Any } ?? NSNull() } }
        if t.function[1 ..< (L - 1)].contains(where: { $0.contains { $0 != 0 } }) {
            out["function"] = decodeFunction(t.function, interproThreshold: params.double("annotation_threshold", default: 0.1),
                                             keywordThreshold: params.double("keyword_threshold", default: 0.5))
        }
        if !t.structure[1 ..< (L - 1)].contains(ESM3Vocab.structureMask) { out["structure_tokens"] = Array(t.structure[1 ..< (L - 1)]) }
        if let structure = structureOutput(t, capture: capture) {
            let letters = Array(SequenceTokens.decode(t.sequence)).map { $0 == "_" ? Character("X") : $0 }
            out["pdb"] = PDB.write(sequence: letters, atoms: structure.atoms, plddt: structure.plddt, title: "ESM3 open (esm3-sm-open-v1)")
            if let plddt = structure.plddt {
                out["plddt"] = plddt
                out["mean_plddt"] = plddt.reduce(0, +) / Double(plddt.count)
            }
            if let ptm = structure.ptm { out["ptm"] = ptm }
            if params.bool("return_pae", default: false), let pae = structure.pae { out["predicted_aligned_error"] = Self.rows(pae) }
        }
        out["generated"] = Self.trackNames.filter(generated.contains)
        return out
    }

    func generate(_ params: Params) throws -> [String: Any] {
        var p = try prompt(params)
        let tracks: [String]
        if let list = params.raw["tracks"] as? [String], !list.isEmpty { tracks = list } else { tracks = [try params.string("track")] }
        guard tracks.allSatisfy(Self.trackNames.contains) else { throw RunnerError.invalid("track must be one of \(Self.trackNames.joined(separator: ", ")).") }
        let s = try GenerationSettings(params, defaults: (1.0, true, "random"), defaultSteps: 20)
        var rng = SeededGenerator(seed: s.seed)
        // condition_on_coordinates_only (default): with coordinates in the prompt, the structure track starts masked and
        // the model reads the coordinates through geometric attention.
        let coordinatesOnly = params.bool("condition_on_coordinates_only", default: true)
        func start(_ tracks: Tracks) -> Tracks {
            var t = tracks
            if coordinatesOnly, t.hasCoordinates {
                for i in 1 ..< (t.length - 1) { t.structure[i] = ESM3Vocab.structureMask }
            }
            return t
        }
        // Each track's masked positions change only when that track is generated, so the plan holds before any pass.
        try RequestLimits.passes(tracks.map { Self.plannedSteps($0, start(p.tracks), s) })
        if params.bool("return_pae", default: false) { try RequestLimits.reply(p.length * p.length, "return_pae for \(p.length) residues") }
        var stepsRun: [String: Int] = [:]
        for track in tracks {
            var t = start(p.tracks)
            stepsRun[track] = try iterate(track, &t, s, rng: &rng, params: params)
            // Only the generated track changes (iterative_sampling_tokens restores the others).
            switch track {
            case "sequence": p.tracks.sequence = t.sequence
            case "structure": p.tracks.structure = t.structure
            case "secondary_structure": p.tracks.ss8 = t.ss8
            case "sasa": p.tracks.sasa = t.sasa
            default: p.tracks.function = t.function
            }
            if capture(params) != nil { save(capture(params)! + ".tokens." + track, ["tokens": trackTokens(p.tracks, track)]) }
        }
        var out = describe(p.tracks, generated: Set(tracks), params: params, capture: capture(params).map { $0 + ".structure" })
        out["seed"] = Int(s.seed)
        out["steps"] = tracks.map { stepsRun[$0] ?? 0 }
        return out
    }

    func capture(_ params: Params) -> String? { captureDir == nil ? nil : (params.raw["capture"] as? String).flatMap(CaptureName.sanitized) }

    func trackTokens(_ t: Tracks, _ track: String) -> MLXArray {
        switch track {
        case "sequence": return MLXArray(t.sequence.map(Int32.init))
        case "structure": return MLXArray(t.structureInput.map(Int32.init))
        case "secondary_structure": return MLXArray(t.ss8.map(Int32.init))
        case "sasa": return MLXArray(t.sasa.map(Int32.init))
        default: return MLXArray(t.function.flatMap { $0.map(Int32.init) }, [t.length, 8])
        }
    }

    func fold(_ params: Params) throws -> [String: Any] {
        var p = try prompt(params, sequenceRequired: true, allowMasks: false)
        if params.bool("return_pae", default: false) { try RequestLimits.reply(p.length * p.length, "return_pae for \(p.length) residues") }
        let s = try GenerationSettings(params, defaults: (1.0, true, "random"))
        var rng = SeededGenerator(seed: s.seed)
        let steps = try iterate("structure", &p.tracks, s, rng: &rng, params: params)
        if let capture = capture(params) { save(capture + ".tokens.structure", ["tokens": trackTokens(p.tracks, "structure")]) }
        guard let structure = structureOutput(p.tracks, capture: capture(params).map { $0 + ".structure" }), let plddt = structure.plddt, let ptm = structure.ptm else {
            throw RunnerError.failed("The structure track was not fully generated.")
        }
        guard let outDir = (params.raw["out_dir"] as? String).map({ URL(fileURLWithPath: $0) }) else { throw RunnerError.invalid("fold needs out_dir (Arsenal sets it).") }
        let url = outDir.appendingPathComponent("structure.pdb")
        let letters = Array(SequenceTokens.decode(p.tracks.sequence))
        try PDB.write(sequence: letters, atoms: structure.atoms, plddt: plddt, title: "ESM3 open (esm3-sm-open-v1) fold").write(to: url, atomically: true, encoding: .utf8)
        var out: [String: Any] = [
            "files": [["id": "structure", "path": url.path, "mime": "chemical/x-pdb", "label": "Predicted backbone (PDB)", "bytes": try Files.size(url), "sha256": try Files.sha256(url)]],
            "sequence": String(letters), "length": p.length, "plddt": plddt, "mean_plddt": plddt.reduce(0, +) / Double(plddt.count), "ptm": ptm,
            "structure_tokens": Array(p.tracks.structure[1 ... p.length]), "seed": Int(s.seed), "steps": steps,
        ]
        if params.bool("return_pae", default: false), let pae = structure.pae { out["predicted_aligned_error"] = Self.rows(pae) }
        return out
    }

    func inverseFold(_ params: Params) throws -> [String: Any] {
        guard params.raw["structure"] != nil else { throw RunnerError.invalid("inverse_fold needs a structure (a PDB file).") }
        var raw = params.raw
        raw["structure_sequence"] = false
        raw.removeValue(forKey: "sequence")
        let local = Params(raw, cancelled: { (try? params.checkCancelled()) == nil })
        var p = try prompt(local)
        let s = try GenerationSettings(local, defaults: (1.0, true, "random"))
        var rng = SeededGenerator(seed: s.seed)
        if local.bool("condition_on_coordinates_only", default: true) {
            for i in 1 ... p.length { p.tracks.structure[i] = ESM3Vocab.structureMask }
        }
        let steps = try iterate("sequence", &p.tracks, s, rng: &rng, params: local)
        let designed = SequenceTokens.decode(p.tracks.sequence)
        var out: [String: Any] = ["sequence": designed, "length": p.length, "seed": Int(s.seed), "steps": steps]
        if let source = p.source {
            let range = (raw["structure_residues"] as? [Any]).flatMap { ($0.first as? NSNumber)?.intValue } ?? 1
            let native = String(source.sequence[(range - 1) ..< Swift.min(source.count, range - 1 + p.length)])
            out["native_sequence"] = native
            let same = zip(native, designed).filter { $0 == $1 }.count
            out["recovery"] = Double(same) / Double(Swift.max(1, native.count))
            out["chain"] = source.chain
        }
        return out
    }

    func predictFunction(_ params: Params) throws -> [String: Any] {
        var p = try prompt(params)
        let s = try GenerationSettings(params, defaults: (1.0, true, "random"))
        var rng = SeededGenerator(seed: s.seed)
        var t = p.tracks
        if params.bool("condition_on_coordinates_only", default: true), t.hasCoordinates {
            for i in 1 ..< (t.length - 1) { t.structure[i] = ESM3Vocab.structureMask }
        }
        for i in 1 ..< (t.length - 1) { t.function[i] = [Int](repeating: 0, count: 8) }
        let steps = try iterate("function", &t, s, rng: &rng, params: params)
        p.tracks.function = t.function
        if let capture = capture(params) { save(capture + ".tokens.function", ["tokens": trackTokens(p.tracks, "function")]) }
        let annotations = decodeFunction(p.tracks.function, interproThreshold: params.double("annotation_threshold", default: 0.1),
                                         keywordThreshold: params.double("keyword_threshold", default: 0.5), capture: capture(params).map { $0 + ".function_decoder" })
        let keywordsFound = Array(Set(annotations.compactMap { $0["kind"] as? String == "keyword" ? $0["label"] as? String : nil })).sorted()
        let interpro = Array(Set(annotations.compactMap { $0["interpro"] as? String })).sorted()
        return ["annotations": annotations, "keywords": keywordsFound, "interpro": interpro, "length": p.length,
                "sequence": SequenceTokens.decode(p.tracks.sequence), "seed": Int(s.seed), "steps": steps]
    }

    func encodeStructure(_ params: Params) throws -> [String: Any] {
        let url = try params.file("structure")
        let backbone = try PDB.parse(try PDB.read(url), chain: (params.raw["chain"] as? String).flatMap { $0.isEmpty ? nil : $0 },
                                     maxResidues: Self.maxResidues)
        try backbone.requireBackbone()
        var taps: [String: MLXArray] = [:]
        let (tokens, latent) = encoder.encode(backbone.atoms, tap: capture(params) == nil ? nil : { taps[$0] = $1 })
        if let capture = capture(params) { save(capture + ".encoder", taps.merging(["latent": latent, "tokens": MLXArray(tokens.map(Int32.init))]) { $1 }) }
        var out: [String: Any] = ["structure_tokens": tokens, "sequence": String(backbone.sequence), "length": backbone.count, "chain": backbone.chain,
                                  "missing_backbone": backbone.atoms.enumerated().filter { $0.element.contains { $0 == nil } }.map { $0.offset + 1 }]
        if params.bool("decode", default: false) {
            var t = Tracks.empty(length: backbone.count)
            for (i, token) in tokens.enumerated() { t.structure[i + 1] = token }
            let letters = backbone.sequence
            if let structure = structureOutput(t, capture: capture(params).map { $0 + ".structure" }), let plddt = structure.plddt {
                out["pdb"] = PDB.write(sequence: letters, atoms: structure.atoms, plddt: plddt, title: "ESM3 open structure tokens, decoded")
                out["plddt"] = plddt
                out["ptm"] = structure.ptm ?? 0
            }
        }
        return out
    }
}
