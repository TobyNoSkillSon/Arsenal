// SPDX-License-Identifier: MIT
// Request-path hardening (weight-free): the shared-prefix cut, the parser's repeated keys, and the picture files the
// model accepts. Each test pins a fixed bug.
@testable import ElephantEngine
import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@Suite struct HardeningTests {
    // MARK: shared-prefix cut (L11)

    @Test func prefixEndsAtTheQuestionEvenWhenTheStateHoldsTheMark() throws {
        let plain = try Prompt.prefixText(state: .string("the user pressed save"))
        #expect(plain.hasSuffix("the user pressed save\n</state>\n\n"))
        // A state quoting the literal mark: the cut stays after the state block, not inside it.
        let quoting = try Prompt.prefixText(state: .string("log line: QEV_PREFIX_MARK seen"))
        #expect(quoting.hasSuffix("log line: QEV_PREFIX_MARK seen\n</state>\n\n"))
        // Media placed after the quoted mark stays in the prefix, so the layout never loses its vision block.
        let media = try Prompt.prefixText(state: .string("QEV_PREFIX_MARK then <image:1>"), media: [.image])
        #expect(media.contains(Prompt.visionBlock) && media.hasSuffix("</state>\n\n"))
    }

    // MARK: repeated keys (L12)

    @Test func repeatedKeysKeepTheLastValueAtTheFirstPlace() throws {
        let v = try PyJSONParser.parse(Data(#"{"b":1,"a":2,"b":3,"e\#u{301}":4,"\#u{E9}":5,"a":6}"#.utf8))
        let fields = try #require(v.fields)
        #expect(fields.map(\.0).map { Array($0.utf8) } == ["b", "a", "e\u{301}", "\u{E9}"].map { Array($0.utf8) })
        #expect(fields.map { $0.1.compact() } == ["3", "6", "4", "5"])
    }

    /// 60,000 distinct keys: the old linear lookup per key took minutes (quadratic); the indexed lookup is linear.
    @Test func manyKeysParseInLinearTime() throws {
        let n = 60_000
        var text = "{"
        for i in 0..<n { text += (i == 0 ? "" : ",") + "\"k\(i)\":\(i)" }
        text += ",\"k0\":-1}"
        let started = Date()
        let v = try PyJSONParser.parse(Data(text.utf8))
        let seconds = Date().timeIntervalSince(started)
        #expect(v.fields?.count == n && v.fields?.first?.1.compact() == "-1" && v.fields?.last?.0 == "k\(n - 1)")
        #expect(seconds < 10, "parse took \(seconds) s")
    }

    // MARK: picture files (L12)

    static func picture(_ type: UTType, in folder: URL, name: String) -> URL? {
        let width = 64, height = 48
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        for y in 0..<height {
            for x in 0..<width {
                context.setFillColor(red: CGFloat(x) / 64, green: CGFloat(y) / 48, blue: CGFloat((x * y) % 7) / 7, alpha: 1)
                context.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        let url = folder.appendingPathComponent(name)
        guard let image = context.makeImage(), let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        return CGImageDestinationFinalize(destination) ? url : nil
    }

    func folder() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("elephant-hardening-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func onlyPNGAndJPEGFilesAreAccepted() throws {
        let dir = try folder()
        defer { try? FileManager.default.removeItem(at: dir) }
        for (type, name) in [(UTType.png, "a.png"), (.jpeg, "a.jpg")] {
            let url = try #require(Self.picture(type, in: dir, name: name))
            #expect(try ImageInput.probe(url).width == 64)
            #expect(try ImageInput.load(url).bytes.count == 64 * 48 * 3)
        }
        // TIFF, GIF and BMP always encode on macOS; HEIC where this Mac has the encoder.
        var refused = 0
        for (type, name) in [(UTType.tiff, "a.tiff"), (.gif, "a.gif"), (.bmp, "a.bmp"), (.heic, "a.heic")] {
            guard let url = Self.picture(type, in: dir, name: name) else { continue }
            #expect(throws: PromptError.self, "\(name)") { try ImageInput.probe(url) }
            // The media list refuses it as a file problem.
            let json = #"[{"type":"image","path":"\#(url.path)"}]"#
            do {
                _ = try Requests.media(try PyJSONParser.parse(Data(json.utf8)), maxItems: 32)
                Issue.record("\(name) accepted")
            } catch let e as Requests.MediaError { #expect(e.code == "invalid_image") }
            refused += 1
        }
        #expect(refused >= 3)
    }

    @Test func truncatedPicturesAreRefusedNotGreyFilled() throws {
        let dir = try folder()
        defer { try? FileManager.default.removeItem(at: dir) }
        for (type, name) in [(UTType.png, "a.png"), (.jpeg, "a.jpg")] {
            let whole = try Data(contentsOf: try #require(Self.picture(type, in: dir, name: name)))
            for fraction in [0.5, 0.9] {
                let cut = dir.appendingPathComponent("cut-\(Int(fraction * 100))-\(name)")
                try whole.prefix(Int(Double(whole.count) * fraction)).write(to: cut)
                // The header still reads, so the request passes its size checks; the decode refuses the file.
                #expect(try ImageInput.probe(cut).width == 64)
                #expect(throws: PromptError.self, "\(cut.lastPathComponent)") { try ImageInput.load(cut) }
            }
        }
    }
}
