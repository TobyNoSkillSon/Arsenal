// SPDX-License-Identifier: MIT
import Foundation
import Testing

@testable import ArsenalCore

/// The esm3-sm-open-v1 kit folder, checked without running the model: its declarations and pins, its embedded kind's
/// eight operations against the runner's own list, its schemas' bounds against the runner's constants, the golden
/// (counts first, then the recorded parity numbers against the frozen bounds) and the weight-free parts of the port,
/// compiled on their own with Tests/KitModels/esm3-sm-open-v1/main.swift and run.
struct Esm3FolderTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private var folder: URL { repo.appendingPathComponent("models/esm3-sm-open-v1") }
    /// The runner sources, bundled into the native worker since 2 Nov (Workers/Kit).
    private var sources: URL { repo.appendingPathComponent("Workers/Kit/Sources/KitEsm3SmOpenV1") }

    private func json(_ path: String) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent(path))) as? [String: Any])
    }

    private func source(_ name: String) throws -> String {
        try String(contentsOf: sources.appendingPathComponent("\(name).swift"), encoding: .utf8)
    }

    @Test func manifestDeclaresEverySourceFileAndTheMeasuredParameterCount() throws {
        let manifest = try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8))
        #expect(manifest.id == "esm3-sm-open-v1" && manifest.kind == "local.protein" && manifest.kindFile == nil)
        #expect(manifest.weights.repo == "biohub/esm3-sm-open-v1" && manifest.weights.revision == "47f0545b2b6daf26a93439a3cd610f4f7f3d5478")
        // 2,701,421,969 BF16 values across the four archives, read from their pickles' persistent ids.
        #expect(manifest.params == "2.70B" && manifest.weights.dtype == "bf16")
        #expect(manifest.derive?["fp16"] == "cast:arsenal-kit-half-v1", "the four .pth and two NumPy files are converted at Get")
        let files = manifest.weights.files
        #expect(files.count == 10)
        for name in [
            "data/weights/esm3_sm_open_v1.pth", "data/weights/esm3_structure_encoder_v0.pth", "data/weights/esm3_structure_decoder_v0.pth",
            "data/weights/esm3_function_decoder_v0.pth", "data/keyword_vocabulary_safety_filtered_58641.txt",
            "data/keyword_idf_safety_filtered_58641.npy", "data/hyperplanes_8bit_58641.npz", "data/interpro_29026_to_keywords_58641.csv",
            "data/entry_list_safety_29026.list", "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv"
        ] {
            #expect(files.contains(name), "\(name) is pinned")
            #expect(manifest.weights.sha256?[name]?.count == 64 && (manifest.weights.bytes?[name] ?? 0) > 0, "\(name) has a hash and a size")
        }
        #expect(manifest.weights.sha256?["data/weights/esm3_sm_open_v1.pth"] == "5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9")
        #expect(manifest.licence.spdx == "MIT" && manifest.licence.commercial == true)
        #expect(manifest.runners.map(\.name) == ["kit"])
        #expect(manifest.languages == ["mul"], "the model reads no human language")
    }

    @Test func everyDeclaredOperationIsTheRunnersOwnListWithBothSchemas() throws {
        let manifest = try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8))
        #expect(manifest.runners.map(\.worker) == ["kit-esm3-sm-open-v1"])
        let spec = try #require(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).first { $0.id == manifest.kind })
        #expect(spec.id == "local.protein" && spec.primaryOperation == "embed_sequence")
        let names = spec.operations.map(\.name)
        let expected = ["embed_sequence", "predict_masked", "score_variants", "generate", "fold", "inverse_fold", "predict_function", "encode_structure"]
        #expect(names.sorted() == expected.sorted(), "the kind declares every op: \(names)")
        // The service offers a kind's default ops when the folder lists none: all eight must be offered, not only the
        // primary one (the service answered 404 for the other seven until 2 Nov).
        #expect(manifest.resolvedOperations(kind: spec).map(\.name).sorted() == expected.sorted())
        // Model.swift's modelOps is what the runner answers; folder_check compares them too, so keep both in step here.
        let model = try source("Model")
        let listed = try #require(model.range(of: "let modelOps = [")).upperBound
        let ops = model[listed...].prefix(while: { $0 != "]" }).split(separator: ",").map { $0.trimmingCharacters(in: CharacterSet(charactersIn: " \"")) }
        #expect(ops.sorted() == expected.sorted(), "Model.swift answers exactly the declared ops: \(ops)")
        for op in spec.operations {
            #expect(op.input == "ops/\(op.name).input.json" && op.outputSchema == "ops/\(op.name).output.json", "\(op.name) names both schemas")
            for side in ["input", "output"] {
                let schema = try JSONSerialization.jsonObject(with: Data(contentsOf: repo.appendingPathComponent("Resources/kinds/local.protein/ops/\(op.name).\(side).json")))
                #expect((schema as? [String: Any])?["type"] as? String == "object", "\(op.name).\(side) is an object schema")
            }
            // fold is the only op that returns files.
            #expect(op.output == (op.name == "fold" ? "file" : "json"), "\(op.name) output transport")
        }
        // The folder keeps a copy of each schema beside model.toml, byte for byte, as Arsenal resolves either path.
        for op in expected {
            for side in ["input", "output"] {
                #expect(
                    try Data(contentsOf: folder.appendingPathComponent("ops/\(op).\(side).json"))
                        == Data(contentsOf: repo.appendingPathComponent("Resources/kinds/local.protein/ops/\(op).\(side).json")), "\(op).\(side)")
            }
        }
    }

    @Test func schemaBoundsMatchTheRunnersConstants() throws {
        let model = try source("Model")
        #expect(model.contains("static let maxResidues = 1024"))
        for op in ["predict_masked", "generate", "fold", "inverse_fold", "predict_function"] {
            let properties = try #require(try json("ops/\(op).input.json")["properties"] as? [String: [String: Any]])
            if let length = properties["length"] {
                #expect(length["maximum"] as? Int == 1024 && length["minimum"] as? Int == 1, "\(op) length bounds")
            }
            if let steps = properties["num_steps"] {
                #expect(steps["maximum"] as? Int == 4096 && steps["minimum"] as? Int == 1, "\(op) num_steps bounds")
            }
            if let schedule = properties["schedule"] {
                #expect(schedule["enum"] as? [String] == ["cosine", "linear"], "\(op) schedule values")
            }
        }
        // Every file input is annotated, or Arsenal would not accept a picked file.
        for op in ["inverse_fold", "encode_structure", "predict_masked", "generate", "score_variants"] {
            if let structure = (try json("ops/\(op).input.json")["properties"] as? [String: [String: Any]])?["structure"] {
                #expect(structure["x-arsenal-file"] != nil, "\(op) structure input is annotated")
            }
        }
        // Release review (2 Nov): an empty variant is refused, and the reply and pass limits the schemas state are the runner's.
        let variants = try #require((try json("ops/score_variants.input.json")["properties"] as? [String: [String: Any]])?["variants"])
        #expect((variants["items"] as? [String: Any])?["minLength"] as? Int == 1, "an empty variant is refused by the schema too")
        let tokens = try source("Tokens")
        #expect(tokens.contains("static let replyNumbers = 640_000") && tokens.contains("static let generationPasses = 1024"))
        for (op, key, phrase) in [
            ("embed_sequence", "per_residue", "415 residues"), ("fold", "return_pae", "800 residues"), ("generate", "return_pae", "800 residues"),
            ("predict_masked", "return_logits", "640,000 numbers"), ("generate", "num_steps", "1,024 steps")
        ] {
            let property = (try json("ops/\(op).input.json")["properties"] as? [String: [String: Any]])?[key]
            #expect((property?["description"] as? String)?.contains(phrase) == true, "\(op).\(key) states \(phrase)")
        }
        let sampling = try source("Sampling")
        #expect(sampling.contains("p_none_threshold") || sampling.contains("pNone > 0.05"), "the function track's <none> rule keeps the maker's 0.05 threshold")
    }

    @Test func goldenRecordsTheFrozenBoundsAndTheReferenceFloor() throws {
        let golden = try json("gate/golden/golden.json")
        #expect(golden["model"] as? String == "esm3-sm-open-v1")
        let reference = try #require(golden["reference"] as? [String: Any])
        #expect((reference["code"] as? String)?.contains("8f0834cbc350aeaf507f755475873f7c61fa44a1") == true, "the reference names the pinned maker commit")
        #expect(reference["device"] as? String == "mps", "the reference ran on the GPU")
        let components = try #require(golden["components"] as? [String: [String: Any]])
        // Counts first: the tokens of every track, four blocks of both cases, every head, the encoder, decoder,
        // function decoder and one generation step.
        #expect(components.count == 57, "every component is recorded: \(components.count)")
        for name in [
            "A.tokens.sequence", "B.tokens.structure", "B.tokens.function", "A.embed_in", "B.block.47", "A.embeddings",
            "B.logits.function", "C.encoder.tokens", "D.plddt", "D.ptm", "E.interpro_logits", "F1.sequence", "G.annotations"
        ] {
            #expect(components[name] != nil, "\(name) is recorded")
        }
        for name in ["A.tokens.sequence", "B.tokens.sequence", "B.tokens.secondary_structure", "B.tokens.sasa", "B.tokens.function"] {
            #expect(components[name]?["mismatches"] as? Int == 0, "\(name) is identical to the maker's")
        }
        #expect(components["C.encoder.tokens"]?["agreement"] as? Double == 1.0, "the structure encoder reproduces the maker's tokens exactly")
        #expect(components["F1.sequence"]?["passed"] as? Bool == true, "one argmax step gives the maker's sequence")
        #expect(components["G.annotations"]?["passed"] as? Bool == true, "the decoded function annotations are identical")
        for name in ["A.embed_in", "A.block.0", "A.block.47", "A.embeddings", "A.logits.sequence", "B.logits.residue_annotations", "E.keyword_logits"] {
            let row = try #require(components[name])
            let measured = try #require(row["rel_l2"] as? Double), bound = try #require(row["bound_rel_l2"] as? Double)
            #expect(measured <= bound, "\(name): relative L2 \(measured) within \(bound)")
            #expect(row["passed"] as? Bool == true, "\(name)")
        }
        // The six misses stay misses against the bf16 GPU reference; the float32 diagnostic is recorded beside them.
        let verdict = try #require(golden["verdict"] as? [String: Any])
        let missed = try #require(verdict["missed"] as? [String: String])
        #expect(Set(missed.keys) == ["A.argmax.structure", "F2.structure_tokens", "F2.ca_rmsd", "F2.plddt", "F2.ptm", "D.backbone"])
        for (name, note) in missed { #expect(note.hasPrefix("a miss against the bf16 GPU reference"), "\(name) is recorded as a miss") }
        #expect(components["D.backbone"]?["passed"] as? Bool == false, "the backbone miss is not hidden")
        let probe = try #require(golden["reference_rounding_floor"] as? [String: [String: Any]])
        for track in ["structure", "fold_step_structure_tokens"] {
            let port = try #require(probe[track]?["port_fp16_vs_maker_fp32_argmax"] as? Double)
            let maker = try #require(probe[track]?["maker_bf16_vs_maker_fp32_argmax"] as? Double)
            #expect(port >= 0.9 && port > maker, "\(track): the port agrees with the maker's float32 (\(port)) more than its own bf16 run does (\(maker))")
        }
        #expect((golden["precision"] as? [String: Any])?["selected"] as? String == "fp16")
        let gate = try String(contentsOf: folder.appendingPathComponent("gate/gate.toml"), encoding: .utf8)
        #expect(gate.contains("precision = \"fp16\"") && gate.contains("evidence = \"gate/golden/golden.json\""))
    }

    /// Review item 8: ESM3's eight ops live in its own kind (signed in Resources/kinds since 2 Nov), so ESM-2 and the
    /// other `local.science` models keep theirs.
    @Test func esm3HasItsOwnKindAndLeavesTheScienceKindAlone() throws {
        let kinds = repo.appendingPathComponent("Resources/kinds")
        #expect(!FileManager.default.fileExists(atPath: folder.appendingPathComponent("kinds").path))
        let kind = try String(contentsOf: kinds.appendingPathComponent("local.protein.toml"), encoding: .utf8)
        #expect(kind.contains("id = \"local.protein\""))
        #expect(try String(contentsOf: repo.appendingPathComponent("models/esm2-t33-650m/model.toml"), encoding: .utf8).contains("kind = \"local.science\""))
        // The shared embed_sequence semantics are kept: same request and output fields as ESM-2's.
        for side in ["input", "output"] {
            let mine = try #require(
                JSONSerialization.jsonObject(with: Data(contentsOf: kinds.appendingPathComponent("local.protein/ops/embed_sequence.\(side).json"))) as? [String: Any])
            let theirs = try #require(
                JSONSerialization.jsonObject(with: Data(contentsOf: kinds.appendingPathComponent("local.science/ops/embed_sequence.\(side).json"))) as? [String: Any])
            #expect(Set((mine["properties"] as? [String: Any] ?? [:]).keys) == Set((theirs["properties"] as? [String: Any] ?? [:]).keys), "embed_sequence \(side) fields")
        }
    }

    /// Review items 12 and 13: the schemas state the runner's defaults and limits, and declare structure_sequence.
    @Test func schemasDeclareTheRunnersDefaults() throws {
        let model = try source("Model")
        // GenerationSettings: temperature 1, annealing on, top_p 1, cosine, random; generate's 20 steps.
        #expect(model.contains("defaults: (1.0, true, \"random\"), defaultSteps: 20"))
        #expect(model.contains("params.double(\"top_p\", default: 1)") && model.contains("params.string(\"schedule\", default: \"cosine\")"))
        for op in ["generate", "fold", "inverse_fold", "predict_function"] {
            let properties = try #require(try json("ops/\(op).input.json")["properties"] as? [String: [String: Any]])
            #expect(properties["temperature"]?["default"] as? Double == 1.0, "\(op) temperature default")
            #expect(properties["temperature_annealing"]?["default"] as? Bool == true, "\(op) annealing default")
            #expect(properties["top_p"]?["default"] as? Double == 1.0, "\(op) top_p default")
            #expect(properties["schedule"]?["default"] as? String == "cosine", "\(op) schedule default")
            #expect(properties["strategy"]?["default"] as? String == "random", "\(op) strategy default")
            let steps = try #require(properties["num_steps"]?["description"] as? String)
            #expect(steps.contains(op == "generate" ? "Default 20" : "residues / 16"), "\(op) states its step default")
        }
        for op in ["predict_masked", "generate", "predict_function"] {
            let properties = try #require(try json("ops/\(op).input.json")["properties"] as? [String: [String: Any]])
            #expect(properties["structure_sequence"]?["default"] as? Bool == true, "\(op) declares structure_sequence")
        }
        // One limit everywhere: Model.maxResidues.
        for op in ["embed_sequence", "predict_masked", "score_variants", "generate", "fold"] {
            let text = try String(contentsOf: folder.appendingPathComponent("ops/\(op).input.json"), encoding: .utf8)
            #expect(!text.contains("1,022") && text.contains("1,024"), "\(op) states the 1,024-residue limit")
        }
        let output = try String(contentsOf: folder.appendingPathComponent("ops/embed_sequence.output.json"), encoding: .utf8)
        #expect(output.contains("before the final LayerNorm"), "pooling is documented as the pre-LayerNorm states")
    }

    /// Review item 9: every component of the golden has a row in the parity log, with the golden's verdict, and the
    /// structure-decoder backbone miss is stated as a miss.
    @Test func parityLogTablesComeFromTheGolden() throws {
        let golden = try json("gate/golden/golden.json")
        let components = try #require(golden["components"] as? [String: [String: Any]])
        let log = try String(contentsOf: folder.appendingPathComponent("log/002-parity.md"), encoding: .utf8)
        let readme = try String(contentsOf: folder.appendingPathComponent("README.md"), encoding: .utf8)
        for (name, row) in components {
            let verdict = (row["passed"] as? Bool) == true ? "pass" : "miss"
            #expect(log.contains("| `\(name)` |") && log.contains("| `\(name)` | \(verdict) |"), "\(name) row with verdict \(verdict)")
        }
        let misses = components.filter { ($0.value["passed"] as? Bool) != true }.map(\.key).sorted()
        #expect(misses.contains("D.backbone"))
        #expect(readme.contains("D.backbone") && readme.contains("\(misses.count) missed"), "the README names the backbone miss and the miss count")
    }

    /// The tokenizers, the function tokenizer's TF-IDF and LSH, the sampling rules, the schedule, the backbone geometry
    /// and PDB reading and writing: compiled from the runner's own sources with no MLX and no weights, then run.
    @Test func weightFreePiecesFollowTheMaker() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("esm3-tokens-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        func run(_ tool: String, _ arguments: [String]) throws -> (Int32, String) {
            let process = Process(), pipe = Pipe()
            process.executableURL = URL(fileURLWithPath: tool)
            process.arguments = arguments
            process.standardOutput = pipe
            process.standardError = pipe
            try process.run()
            let output = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return (process.terminationStatus, String(decoding: output, as: UTF8.self))
        }
        let binary = temporary.appendingPathComponent("check").path
        let files =
            ["Tokens", "Protein", "Sampling"].map { sources.appendingPathComponent("\($0).swift").path }
            + [repo.appendingPathComponent("Tests/KitModels/esm3-sm-open-v1/main.swift").path]
        let compile = try run("/usr/bin/xcrun", ["swiftc", "-Onone"] + files + ["-o", binary])
        try #require(compile.0 == 0, "\(compile.1)")
        // The fixture crop of the maker's own data/1utn.pdb (lab evidence; the PDB checks are skipped without it).
        let fixture = repo.appendingPathComponent("lab/evidence/esm3-sm-open-v1/1utn-crop64.pdb")
        let result = try run(binary, FileManager.default.fileExists(atPath: fixture.path) ? [fixture.path] : [])
        #expect(result.0 == 0, "\(result.1)")
        #expect(result.1.hasPrefix("ok "), "\(result.1)")
    }
}
