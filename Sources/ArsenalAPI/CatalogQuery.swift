// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

func catalogQuery(_ query: [String: String]) throws -> CatalogQuery {
    var result = CatalogQuery(text: query["q"] ?? "")
    func list(_ key: String) -> [String] { (query[key] ?? "").split(separator: ",").map(String.init) }
    func bool(_ key: String) throws -> Bool? {
        guard let raw = query[key] else { return nil }
        guard ["true", "false"].contains(raw) else { throw HTTPFailure(422, "\(key) must be true or false.") }
        return raw == "true"
    }
    result.kinds = list("kind"); result.licences = list("licence"); result.languages = list("language"); result.tasks = list("task"); result.dtypes = list("dtype")
    result.modalitiesIn = list("modality_in"); result.modalitiesOut = list("modality_out")
    result.commercial = try bool("commercial"); result.suggested = try bool("suggested"); result.added = try bool("added"); result.installed = try bool("installed")
    result.loaded = try bool("loaded"); result.fitsRAM = try bool("fits_ram")
    if let raw = query["max_size_bytes"] {
        guard let bytes = Int64(raw), bytes >= 0 else { throw HTTPFailure(422, "max_size_bytes must be a nonnegative integer.") }; result.maxSizeBytes = bytes
    }
    if let raw = query["min_size_bytes"] {
        guard let bytes = Int64(raw), bytes >= 0 else { throw HTTPFailure(422, "min_size_bytes must be a nonnegative integer.") }; result.minSizeBytes = bytes
    }
    for (key, minimum) in [("min_ram_mb", true), ("max_ram_mb", false)] {
        if let raw = query[key] {
            guard let value = Double(raw), value.isFinite, value >= 0 else { throw HTTPFailure(422, "\(key) must be a nonnegative finite number.") }
            if minimum { result.minRAMMB = value } else { result.maxRAMMB = value }
        }
    }
    if let min = result.minSizeBytes, let max = result.maxSizeBytes, min > max { throw HTTPFailure(422, "Minimum size must not exceed maximum size.") }
    if let min = result.minRAMMB, let max = result.maxRAMMB, min > max { throw HTTPFailure(422, "Minimum RAM must not exceed maximum RAM.") }
    if let raw = query["sort"] { guard let sort = CatalogSort(rawValue: raw) else { throw HTTPFailure(422, "Unknown catalog sort column.") }; result.sort = sort }
    if let order = query["order"] { guard ["asc", "desc"].contains(order) else { throw HTTPFailure(422, "order must be asc or desc.") }; result.descending = order == "desc" }
    return result
}
