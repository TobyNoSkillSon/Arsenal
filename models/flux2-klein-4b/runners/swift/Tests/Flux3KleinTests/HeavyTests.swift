import Foundation
import Testing

/// Heavy tests run model inference or GPU model code: opt-in only, `scripts/test.sh --heavy --filter Flux3`.
extension Trait where Self == ConditionTrait {
    static var heavy: Self { .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_HEAVY_TESTS"] == "1") }
}
