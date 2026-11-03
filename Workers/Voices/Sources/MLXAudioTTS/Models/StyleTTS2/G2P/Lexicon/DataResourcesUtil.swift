import Foundation

/// misaki's gold and silver dictionaries. Vireo: US English reads us_* from the G2P asset, UK English gb_* from the
/// British lexicon asset (hexgrad/misaki); a missing or unreadable file is a load error, never an empty dictionary or
/// the American one under a British label.
final class DataResourcesUtil {
    private init() {}

    enum LoadError: Error, LocalizedError {
        case missing(URL)
        var errorDescription: String? {
            switch self {
            case .missing(let url): return "Kokoro phonemizer dictionary missing or unreadable: \(url.lastPathComponent)"
            }
        }
    }

    static func loadGold(british: Bool, directory: URL) throws -> [String: Any] {
        try loadJSON(filename: british ? "gb_gold" : "us_gold", directory: directory)
    }

    static func loadSilver(british: Bool, directory: URL) throws -> [String: Any] {
        try loadJSON(filename: british ? "gb_silver" : "us_silver", directory: directory)
    }

    private static func loadJSON(filename: String, directory: URL) throws -> [String: Any] {
        let url = directory.appendingPathComponent("\(filename).json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any], !json.isEmpty else {
            throw LoadError.missing(url)
        }
        return json
    }
}
