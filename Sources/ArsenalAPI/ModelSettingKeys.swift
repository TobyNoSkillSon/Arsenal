// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// The settings every model has besides its own declared ones: `dtype`, `keep_hot`, `show_in_menu` and `runner`. MCP's
/// `arsenal_settings(set:)` and HTTP's `POST /arsenal/v1/models/<id>/settings` accept the same keys with the same checks.
enum ModelSettingKeys {
    static let reserved = ["dtype", "keep_hot", "show_in_menu", "runner"]

    /// A reserved value refused before anything is applied. `unsupported`: the value is valid but not usable here.
    struct Refusal: Error, Equatable {
        let unsupported: Bool
        /// Without a leading "Not changed:".
        let text: String
    }

    /// The reserved keys present, in `reserved` order, each checked; the model's own settings are left for the service.
    /// `show_in_menu` travels with the declared settings (`__tools`) in the service's one atomic write.
    static func split(_ values: JSONObject, entry: CatalogEntry) throws(Refusal) -> (declared: JSONObject, reserved: [(String, JSONValue)]) {
        var reserved: [(String, JSONValue)] = []
        for key in Self.reserved { if let value = values[key] { reserved.append((key, value)) } }
        for (key, value) in reserved {
            switch key {
            case "dtype":
                guard let dtype = value.string, entry.dtypes.contains(dtype) else {
                    throw Refusal(unsupported: false, text: "dtype must be one of \(entry.dtypes.joined(separator: ", ")).")
                }
                if let reason = entry.row.dtypeUnavailableReason(dtype) { throw Refusal(unsupported: true, text: "\(dtype) is unavailable: \(reason)") }
            case "runner":
                guard let name = value.string, entry.runners.contains(where: { $0.name == name && $0.eligible }) else {
                    throw Refusal(unsupported: true, text: "runner must be one of \(entry.runners.filter(\.eligible).map(\.name).joined(separator: ", ")).")
                }
            default:
                guard value.bool != nil else { throw Refusal(unsupported: false, text: "\(key) must be true or false.") }
            }
        }
        var declared = values
        for key in Self.reserved { declared[key] = nil }
        if let shown = values["show_in_menu"] { declared["__tools"] = shown; reserved.removeAll { $0.0 == "show_in_menu" } }
        return (declared, reserved)
    }

    /// Applies one checked reserved key (not `show_in_menu`, which `split` moved into the declared write).
    static func apply(_ key: String, _ value: JSONValue, id: String, service: any ModelsService) async throws {
        switch key {
        case "dtype": try await service.setPrecision(id: id, precision: value.string!)
        case "keep_hot": try await service.setKeepHot(id: id, on: value.bool!)
        case "show_in_menu": try await service.setToolsShown(id: id, on: value.bool!)
        default: try await service.setRunner(id: id, runner: value.string)
        }
    }
}
