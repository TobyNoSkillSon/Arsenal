// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// SAM 3.1, SAM Audio Large and TRELLIS.2's DINOv3 encoder come only from Meta's own gated repositories, as FLUX 3 Image
/// Klein comes from Black Forest Labs': the user accepts Meta's terms on Hugging Face and Arsenal downloads with the
/// user's token. No re-hosted or ungated copy is pinned, the terms are shown before Get, and the licence texts ship.
struct MetaGatedModelsTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func load(_ id: String) throws -> ModelManifest {
        try ModelManifest.load(repo.appendingPathComponent("models/\(id)"), kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    }
    /// The model, Meta's gated repository it downloads from, and the pinned revision.
    static let cases: [(id: String, gated: String, revision: String)] = [
        ("sam3.1", "facebook/sam3.1", "daa63191845a41281374e725f4c9e51c7a824460"),
        ("sam-audio-large", "facebook/sam-audio-large", "5f2cd3a9471a08c7282c06036be6893e18de8b70"),
        ("trellis2-4b", "facebook/dinov3-vitl16-pretrain-lvd1689m", "ea8dc2863c51be0a264bab82070e3e8836b02d51"),
        ("idol", "facebook/dinov3-vith16plus-pretrain-lvd1689m", "c807c9eeea853df70aec4069e6f56b28ddc82acc")
    ]

    @Test(arguments: cases.map(\.id)) func downloadsOnlyFromMetasOwnGatedRepository(id: String) throws {
        let c = try #require(Self.cases.first { $0.id == id })
        let model = try load(id)
        let sources = PinnedFilesCheck.baseSources(model)
        let gated = try #require(sources.first { $0.repo == c.gated }, "\(id) does not download from \(c.gated)")
        #expect(gated.revision == c.revision)
        #expect(!gated.files.isEmpty && gated.files.allSatisfy { ModelManifest.isHex(gated.sha256?[$0] ?? "", count: 64) && (gated.bytes?[$0] ?? 0) > 0 })
        // Nothing comes from a re-upload of a Meta model: every Meta-published source is Meta's own repository.
        for source in sources {
            #expect(!source.repo.hasPrefix("mlx-community/") && !source.repo.hasPrefix("camenduru/"), "\(id) pins the copy \(source.repo)")
        }
    }

    @Test(arguments: cases.map(\.id)) func termsAreShownBeforeGet(id: String) throws {
        let c = try #require(Self.cases.first { $0.id == id })
        let model = try load(id)
        #expect(!model.licence.commercial)
        let conditions = try #require(model.licence.conditions)
        #expect(conditions.contains("on Hugging Face") && conditions.contains("Meta approves each request") && conditions.contains("ITAR"))
        #expect(conditions.contains("downloads from \(c.gated) with your") && conditions.contains("token"))
        if id == "trellis2-4b" || id == "idol" {
            #expect(model.licence.spdx == (id == "idol" ? "Apache-2.0" : "MIT") && conditions.contains("DINOv3 License"))
            let dino = try #require(model.selectedAssets.first { $0.repo == c.gated })
            #expect(dino.licence == .spdx("LicenseRef-DINOv3"))
        } else {
            #expect(model.licence.spdx == "LicenseRef-SAM" && model.licence.name == "SAM License")
            #expect(model.licence.url == "https://huggingface.co/\(c.gated)/blob/\(c.revision)/LICENSE")
        }
    }

    /// A refused Get opens Meta's page for the gated repository (and the token page when Arsenal has none) and says what
    /// the user must do; Meta approves each request by hand, so the gate is "manual".
    @Test(arguments: cases.map(\.gated)) func gateOpensMetasPage(gated: String) {
        let noToken = HFGateError(repo: gated, hasToken: false, mode: "manual")
        #expect(noToken.pages.map(\.absoluteString) == ["https://huggingface.co/\(gated)", "https://huggingface.co/settings/tokens"])
        #expect(noToken.errorDescription?.contains("add one in Settings") == true)
        let withToken = HFGateError(repo: gated, hasToken: true, mode: "manual")
        #expect(withToken.pages.map(\.absoluteString) == ["https://huggingface.co/\(gated)"])
        #expect(withToken.errorDescription == "\(gated) is gated on Hugging Face: accept its terms on the model's page, then Get again.")
    }

    /// The pin check before approval lists Meta's repository; a refusal there is the gate, never an error row.
    @Test(arguments: cases.map(\.id)) func pinCheckLeavesTheGateToGet(id: String) async throws {
        let c = try #require(Self.cases.first { $0.id == id })
        let model = try load(id)
        let check = await PinnedFilesCheck.run(model) { repo, revision, _ in
            if repo == c.gated { return .gated(repo: repo) }
            let source = PinnedFilesCheck.baseSources(model).first { $0.repo == repo && $0.revision == revision }
            return .files(Set(source?.files ?? []))
        }
        #expect(check == .gated(repo: c.gated))
        #expect(check.problem == nil && check.note?.contains(c.gated) == true)
    }

    /// Meta's licences require a copy with anything distributed: their full texts ship in the app's Licenses folder
    /// (scripts/assemble.sh copies docs/notices/LICENSE-*), and the notice names all three models.
    @Test func licenceTextsShipWithTheApp() throws {
        let notices = repo.appendingPathComponent("docs/notices")
        let sam = try String(contentsOf: notices.appendingPathComponent("LICENSE-Meta-SAM"), encoding: .utf8)
        #expect(sam.hasPrefix("SAM License") && sam.contains("ITAR"))
        let dino = try Data(contentsOf: notices.appendingPathComponent("LICENSE-Meta-DINOv3.md"))
        #expect(dino == (try Data(contentsOf: repo.appendingPathComponent("models/trellis2-4b/LICENSE-DINOv3.md"))))
        #expect(dino == (try Data(contentsOf: repo.appendingPathComponent("models/idol/LICENSE-DINOv3.md"))))
        #expect(String(decoding: dino, as: UTF8.self).hasPrefix("# DINOv3 License"))
        let notice = try String(contentsOf: notices.appendingPathComponent("meta-gated.md"), encoding: .utf8)
        for c in Self.cases { #expect(notice.contains(c.gated)) }
        #expect(notice.contains("Licenses/LICENSE-Meta-SAM") && notice.contains("Licenses/LICENSE-Meta-DINOv3.md"))
    }

    /// IDOL's FLUX.2 VAE is FLUX.2 klein 4B's (Apache-2.0), never FLUX.2-dev's (non-commercial): with IDOL's BF16 cast
    /// the two are bit-identical (models/idol/results/vae-swap-2026-11-02.md).
    @Test func idolUsesTheApacheKleinVAE() throws {
        let model = try load("idol")
        let vae = try #require(model.selectedAssets.first { $0.id == "idol-flux2-vae" })
        #expect(vae.repo == "black-forest-labs/FLUX.2-klein-4B" && vae.licence == .spdx("Apache-2.0"))
        #expect(!PinnedFilesCheck.baseSources(model).contains { $0.repo == "black-forest-labs/FLUX.2-dev" })
    }
}
