// SPDX-License-Identifier: MIT
import Foundation

public enum MemoryWhenFull: String, Codable, Sendable, CaseIterable { case refuse, queue }
public struct MemorySettings: Codable, Sendable, Equatable {
    public var whenFull: MemoryWhenFull
    public init(whenFull: MemoryWhenFull = .refuse) { self.whenFull = whenFull }
    enum CodingKeys: String, CodingKey { case whenFull = "when_full" }
}
