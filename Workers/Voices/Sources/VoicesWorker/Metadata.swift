// SPDX-License-Identifier: MIT
import Foundation
import VireoCore

struct SpeechSpec: Decodable {
    let id: String
    let tier16_dtype: String
    let tiers_offered: [String]
    let voices: [String]
    let files: [String]
}
enum SpeechCatalog {
    static let all: [SpeechSpec] = {
        struct Catalog: Decodable { let models: [SpeechSpec] }
        return (try? JSONDecoder().decode(Catalog.self, from: Data(contentsOf: Bundle.module.url(forResource: "models", withExtension: "json")!)).models) ?? []
    }()
    static func model(_ id: String) throws -> SpeechSpec {
        guard let spec = all.first(where: { $0.id == id }) else { throw EngineError("Unknown voice model \(id)") }
        return spec
    }
    static func checkPrecision(_ dtype: String, spec: SpeechSpec) throws {
        guard dtype == spec.tier16_dtype || (dtype == "int8" && spec.tiers_offered.contains("8")) || (dtype == "int4" && spec.tiers_offered.contains("4")) else {
            throw EngineError("\(spec.id) does not offer \(dtype)")
        }
    }
}
