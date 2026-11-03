// SPDX-License-Identifier: MIT
import Foundation

public struct MenuSectionSpec: Codable, Sendable, Equatable {
    public var title: String
    public var items: [MenuSpec]
    public init(title: String, items: [MenuSpec]) { self.title = title; self.items = items }
}
/// Accepts legacy [[menu]] actions and v2.2 [[menu.section]] without rewriting any existing folder.
public struct ModelMenu: Codable, Sendable, Equatable {
    public var actions: [MenuSpec]
    public var sections: [MenuSectionSpec]
    public init(from decoder: any Decoder) throws {
        let single = try decoder.singleValueContainer()
        if let legacy = try? single.decode([MenuSpec].self) { actions = legacy; sections = []; return }
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sections = try c.decode([MenuSectionSpec].self, forKey: .section); actions = []
    }
    public func encode(to encoder: any Encoder) throws {
        if sections.isEmpty {
            var c = encoder.singleValueContainer(); try c.encode(actions)
        } else {
            var c = encoder.container(keyedBy: CodingKeys.self); try c.encode(sections, forKey: .section)
        }
    }
    enum CodingKeys: String, CodingKey { case section }
}
