import Foundation
import MLXAudioCore

public final class MisakiTextProcessor: TextProcessor, @unchecked Sendable {
    private var usG2P: EnglishG2P?
    private var gbG2P: EnglishG2P?
    private let lock = NSLock()
    private nonisolated(unsafe) var resourceDirectory: URL?
    private let britishDirectory: URL?
    private let precision: BARTPrecision

    /// `resourceDirectory`: the G2P asset (US dictionaries, BART fallback); `britishDirectory`: misaki's gb_* dictionaries,
    /// without which British English is an error rather than American pronunciation.
    public init(resourceDirectory: URL? = nil, britishDirectory: URL? = nil, precision: BARTPrecision = BARTPrecision()) {
        self.precision = precision
        self.resourceDirectory = resourceDirectory; self.britishDirectory = britishDirectory
    }

    public func prepare() async throws {
        guard resourceDirectory != nil else { throw MisakiError.resourcesNotDownloaded }
    }

    public func process(text: String, language: String?) throws -> String {
        let british = language?.lowercased().contains("gb") == true
        let g2p = try getG2P(british: british)
        let (phonemes, _) = g2p.phonemize(text: text)
        return phonemes
    }

    private func getG2P(british: Bool) throws -> EnglishG2P {
        lock.lock()
        defer { lock.unlock() }
        guard let dir = resourceDirectory else {
            throw MisakiError.resourcesNotDownloaded
        }
        if british {
            if let cached = gbG2P { return cached }
            let g2p = try EnglishG2P(british: true, directory: dir, lexiconDirectory: britishDirectory, precision: precision)
            gbG2P = g2p
            return g2p
        } else {
            if let cached = usG2P { return cached }
            let g2p = try EnglishG2P(british: false, directory: dir, precision: precision)
            usG2P = g2p
            return g2p
        }
    }

    func intakeModel() throws -> BARTModel { try getG2P(british: false).fallback.model }

    enum MisakiError: Error, LocalizedError {
        case resourcesNotDownloaded

        var errorDescription: String? {
            switch self {
            case .resourcesNotDownloaded:
                return "G2P resources not downloaded. Call prepare() first."
            }
        }
    }
}
