// SPDX-License-Identifier: MIT
import Testing
import ArsenalWire

@Test func timingUsesUncachedPrefillAndTokensAfterTheFirst() {
    var tracker = TokenTiming(accepted: 10, scheduled: 12)
    tracker.prefillStarted = 12; tracker.promptTokens = 1000; tracker.cachedTokens = 900
    #expect(tracker.timing().ttft_s == nil && tracker.timing().decode_tps == nil)
    tracker.token(at: 14)
    #expect(tracker.timing().ttft_s == 4)
    #expect(tracker.timing().prefill_tps == 50)
    #expect(tracker.timing().decode_tps == nil && tracker.timing().max_itl_ms == nil)
    tracker.token(at: 14.25); tracker.token(at: 15)
    let timing = tracker.timing(peakMB: 512)
    #expect(timing.queue_s == 2 && timing.decode_tps == 2)
    #expect(timing.max_itl_ms == 750 && timing.peak_mb == 512)
}

@Test func unknownTimingsAreAbsentOnWire() {
    #expect(RequestTiming().json.isEmpty)
    #expect(RequestTiming(peak_mb: 42).json == ["peak_mb": 42])
}
