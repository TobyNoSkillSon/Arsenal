// SPDX-License-Identifier: MIT
import Foundation

extension RealModelsService {
    public func kinds() -> [KindSpec] {
        // An embedded kind repeating a signed one (ModelManifest allows only an identical copy) adds no second group.
        let embedded = localModels.values.compactMap(\.embeddedKind).filter { kind in !specs.contains { $0.id == kind.spec.id } }
        let all = specs + Dictionary(grouping: embedded, by: { $0.spec.id }).values.compactMap { $0.first?.spec }
        return all.sorted { a, b in
            let ia = config.kindOrder.firstIndex(of: a.id) ?? Int.max, ib = config.kindOrder.firstIndex(of: b.id) ?? Int.max
            return ia != ib ? ia < ib : (a.order, a.id) < (b.order, b.id)
        }
    }
    public func persistedKindOrder() async -> [String] { config.kindOrder }
    public func setKindOrder(_ order: [String]) async throws {
        let known = Set(kinds().map(\.id)).union(config.kindOrder)
        guard Set(order).count == order.count, order.allSatisfy({ $0 != "local" && ModelManifest.validID($0) && known.contains($0) }) else {
            throw WorkerFailure(code: "kind_order", message: "Group order must contain unique known or remembered kind IDs.")
        }
        config.kindOrder = order; config.requestedKindOrder = order
        try persist(); changed()
    }
}
