// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// Job files returned as URLs share one per-job artifact limit, separate from the 64 MiB inline cap.
struct OperationArtifactLimitTests {
    @Test func jobFilesAdmitUpToTheArtifactLimitAndNameItWhenExceeded() throws {
        let gib = Int64(1) << 30
        #expect(OperationArtifacts.maxBytes == 64 * 1024 * 1024, "the inline cap is unchanged")
        var total = 0
        try OperationArtifacts.admitArtifact(3 * gib, total: &total) // far above the inline cap, admitted as a job file
        try OperationArtifacts.admitArtifact(gib, total: &total) // exactly the per-job limit
        #expect(total == Int(4 * gib))
        do {
            try OperationArtifacts.admitArtifact(1, total: &total)
            Issue.record("a file past the per-job artifact limit was admitted")
        } catch let failure as WorkerFailure {
            #expect(failure.code == "invalid_output")
            #expect(failure.message == "Job output files exceed the 4 GiB per-job artifact limit.")
        }
    }
}
