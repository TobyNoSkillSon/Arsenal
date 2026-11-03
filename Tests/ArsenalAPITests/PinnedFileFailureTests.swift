// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
@testable import ArsenalAPI

/// A pinned file Hugging Face lacks at its revision reaches HTTP clients as a final failure, never a resumable download.
@Test func pinnedFileMissingIsFinalOverHTTP() {
    let error = HFMissingFileError(repo: "owner/model", revision: String(repeating: "c", count: 40), path: "model.safetensors")
    let failure = serviceHTTPFailure(error, modelID: "local-model")
    #expect(failure.status == 422 && failure.code == "pinned_file_missing")
    #expect(failure.message.hasPrefix("model.safetensors is not in the repo owner/model at revision cccccccccccc"))
    #expect(!failure.message.contains("resume"))
}

/// A Get started in the app or over HTTP that ended on a missing pinned file: MCP following it says not to retry.
@Test func externalGetOnAMissingPinIsNotResumable() {
    let missing = PrecisionState(state: .failed, error: "model.safetensors is not in the repo owner/model at revision cccccccccccc.", phase: "pinned_file_missing")
    let line = APIRouter.mcpFinalGetFailure(missing, name: "Local Model")?.line
    #expect(
        line == "[download] Download of Local Model failed: model.safetensors is not in the repo owner/model at revision cccccccccccc. "
            + "Do not call download again; the model's pins must change.")
    #expect(line?.contains("allow_download") == false)
    let broken = PrecisionState(state: .failed, error: "owner/model has no file a.bin", phase: "localFolderError")
    #expect(APIRouter.mcpFinalGetFailure(broken, name: "Local Model") != nil)
    // Any other failed Get keeps the resume call.
    #expect(APIRouter.mcpFinalGetFailure(PrecisionState(state: .failed, error: "stalled"), name: "Local Model") == nil)
}
