// SPDX-License-Identifier: MIT
import Foundation

/// Semantic release ordering; build metadata does not change precedence.
public struct UpdateVersion: Comparable, Sendable, CustomStringConvertible {
    public let description: String
    private let numbers: [Int]
    private let prerelease: [String]
    public var isPrerelease: Bool { !prerelease.isEmpty }

    public init?(_ text: String) {
        var value = text
        if value.hasPrefix("v") { value.removeFirst() }
        let metadata = value.split(separator: "+", omittingEmptySubsequences: false)
        guard metadata.count <= 2 else { return nil }
        func identifiers(_ text: String) -> [String]? {
            let parts = text.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
            return parts.allSatisfy { !$0.isEmpty && $0.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") } } ? parts : nil
        }
        if metadata.count == 2, identifiers(String(metadata[1])) == nil { return nil }
        let pieces = metadata[0].split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
        let components = pieces[0].split(separator: ".", omittingEmptySubsequences: false)
        guard components.count == 3 else { return nil }
        var numbers: [Int] = []
        for part in components {
            guard !part.isEmpty, part.allSatisfy({ $0.isASCII && $0.isNumber }), part.count == 1 || part.first != "0", let number = Int(part) else { return nil }
            numbers.append(number)
        }
        var prerelease: [String] = []
        if pieces.count == 2 {
            guard let parsed = identifiers(String(pieces[1])) else { return nil }
            for part in parsed where part.allSatisfy(\.isNumber) {
                guard part.count == 1 || part.first != "0" else { return nil }
            }
            prerelease = parsed
        }
        self.numbers = numbers; self.prerelease = prerelease; description = value
    }

    public static func == (lhs: Self, rhs: Self) -> Bool { lhs.numbers == rhs.numbers && lhs.prerelease == rhs.prerelease }
    public static func < (lhs: Self, rhs: Self) -> Bool {
        if lhs.numbers != rhs.numbers { return lhs.numbers.lexicographicallyPrecedes(rhs.numbers) }
        if lhs.prerelease.isEmpty { return false }
        if rhs.prerelease.isEmpty { return true }
        for (left, right) in zip(lhs.prerelease, rhs.prerelease) where left != right {
            let leftNumeric = left.allSatisfy(\.isNumber), rightNumeric = right.allSatisfy(\.isNumber)
            if leftNumeric && rightNumeric { return left.count == right.count ? left < right : left.count < right.count }
            if leftNumeric != rightNumeric { return leftNumeric }
            return left < right
        }
        return lhs.prerelease.count < rhs.prerelease.count
    }
}

public struct UpdateFailure: LocalizedError, Sendable, Equatable {
    public let reason: String
    public init(_ reason: String) { self.reason = reason }
    public var errorDescription: String? { reason }
}

public enum UpdateState: Equatable, Sendable {
    case none
    case available(version: String, notes: String, size: Int64)
    case downloading(progress: Double)
    /// Verified and waiting for the service to drain and block new generation requests.
    case ready
    case installing
    case failed(reason: String)
}
