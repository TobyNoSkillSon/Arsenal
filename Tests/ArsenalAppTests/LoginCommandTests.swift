// SPDX-License-Identifier: MIT
import Testing
@testable import Arsenal

/// `Arsenal login`: arguments are checked before Launch at Login is touched; a bare binary refuses.
@MainActor @Test func loginCommandChecksItsArgumentsAndBundle() {
    #expect(LoginCommand.run([]) == 2)
    #expect(LoginCommand.run(["maybe"]) == 2)
    #expect(LoginCommand.run(["off", "now"]) == 2)
    #expect(LoginCommand.run(["off"], bundleIdentifier: nil) == 1)
}
