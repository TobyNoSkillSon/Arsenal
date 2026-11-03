import Testing
@testable import MusicWorker
struct RegistryTests {
    @Test func matrixEligibility() {
        #expect(MusicConversion.eligible("lm_head.weight", [64, 128]))
        #expect(!MusicConversion.eligible("conv.weight", [64, 3, 128]))
    }
}
