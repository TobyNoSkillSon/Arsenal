// SPDX-License-Identifier: MIT
// perception.detect plumbing shared by the open-vocabulary detectors: the text queries from the request, and the reply.
import Foundation
import MLX

enum Queries {
    /// `labels` (one query each), else `phrase`, else `instruction`, split on "." "," ";" and new lines ("cat. remote control").
    static func from(_ params: Params, limit: Int) throws -> [String] {
        var queries: [String]
        if let labels = params.raw["labels"] as? [String] {
            queries = labels
        } else if let text = (params.raw["phrase"] as? String) ?? (params.raw["instruction"] as? String) {
            queries = text.components(separatedBy: CharacterSet(charactersIn: ".,;\n"))
        } else {
            throw RunnerError.invalid("Give labels (a list of things to find) or a phrase such as \"cat. remote control\".")
        }
        var seen = Set<String>()
        queries = queries.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty && seen.insert($0).inserted }
        guard !queries.isEmpty else { throw RunnerError.invalid("The labels or phrase hold no words to look for.") }
        guard queries.count <= limit else { throw RunnerError.invalid("At most \(limit) labels per request.") }
        return queries
    }
}

struct Detection {
    let label: String
    let score: Float
    /// Corners in the photo's pixels.
    let x0: Float, y0: Float, x1: Float, y1: Float
}

enum Reply {
    /// perception.detect's reply: boxes normalized to the photo (0...1, clipped), best score first, at most 1024.
    static func detections(_ found: [Detection], width: Int, height: Int) -> [String: Any] {
        let w = Float(width), h = Float(height)
        func unit(_ v: Float) -> Double { Double(min(max(v, 0), 1)) }
        let sorted = found.sorted { $0.score > $1.score }.prefix(1024)
        return [
            "detections": sorted.map { d -> [String: Any] in
                [
                    "label": d.label,
                    "score": Double(min(max(d.score, 0), 1)),
                    "box": ["xmin": unit(d.x0 / w), "ymin": unit(d.y0 / h), "xmax": unit(d.x1 / w), "ymax": unit(d.y1 / h)],
                ]
            },
            "coordinate_system": "normalized",
        ]
    }
}
