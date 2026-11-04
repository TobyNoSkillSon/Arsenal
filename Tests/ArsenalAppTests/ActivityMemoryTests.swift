// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal
import ArsenalCore

/// Each card says what its model's memory is for (a card read 8.4 GB for SeedVR2 while each batch took
/// 23–29 GB more): weights, working memory and the job's peak, from C3's `memory`.
@MainActor @Suite(.menuState, .serialized) final class ActivityMemoryTests {
    @Test func theFeedReadsC3sMemoryFromJSONAndFromTheService() throws {
        let json: [String: Any] = [
            "models": [
                [
                    "id": "seedvr2-3b", "kind": "restoration", "memory_mb": 31_700,
                    "memory": ["source": "mlx", "total_mb": 31_700, "weights_mb": 8_438, "working_mb": 21_900, "cache_mb": 790, "peak_mb": 37_900, "mlx_peak_mb": 37_300]
                ],
                ["id": "older", "kind": "chat", "memory_mb": 400]
            ]
        ]
        let feed = ActivityFeed(json: json)
        let memory = try #require(feed.models[0].memory)
        #expect(memory == .init(source: "mlx", totalMB: 31_700, weightsMB: 8_438, workingMB: 21_900, cacheMB: 790, peakMB: 37_900, mlxPeakMB: 37_300))
        #expect(feed.models[1].memory == nil, "a service without C3 memory shows the card as before")
        let snapshot = ActivitySnapshot(models: [
            ModelActivity(
                id: "seedvr2-3b", kind: "restoration", memoryMB: 9_013,
                memory: ModelMemoryActivity(WorkerMemory(footprintMB: 9_013, mlxActiveMB: 8_438, mlxCacheMB: 252, mlxWeightsMB: 8_438)))
        ])
        let served = try #require(ActivityFeed(snapshot).models.first?.memory)
        #expect(served.source == "mlx" && served.weightsMB == 8_438 && served.workingMB == 0 && served.cacheMB == 252 && served.peakMB == nil)
    }

    @Test func theBarAndItsFiguresSayWhatHoldsTheMemory() {
        let seedvr = ActivityFeed.Memory(source: "mlx", totalMB: 31_700, weightsMB: 8_438, workingMB: 21_900, cacheMB: 790, peakMB: 37_900, mlxPeakMB: 37_300)
        #expect(ActivityText.memoryBar(seedvr).map(\.layer) == [.weights, .working, .cache, .outside])
        #expect(ActivityText.memoryBar(seedvr).map(\.mb) == [8_438, 21_900, 790, 572])
        // Working memory's figure includes MLX's cache; the bar draws it apart, lighter.
        #expect(ActivityText.memoryLegend(seedvr).map(\.name) == ["Weights", "Working", "Peak"])
        #expect(ActivityText.memoryLegend(seedvr).map(\.mb) == [8_438, 22_690, 37_900])
        let help = ActivityText.memoryHelp(seedvr)
        #expect(help.contains("Weights 7.9 GB") && help.contains("Peak 35.3 GB") && help.contains("MLX\u{2019}s own peak was 34.7 GB"))
        let arrays = ActivityFeed.Memory(source: "mlx", totalMB: 610, arraysMB: 420, cacheMB: 90)
        #expect(ActivityText.memoryBar(arrays).map(\.layer) == [.weights, .cache, .outside])
        #expect(ActivityText.memoryLegend(arrays).map(\.name) == ["In use", "Cache", "Peak"])
        #expect(ActivityText.memoryLegend(arrays).last?.mb == nil, "no job yet: the peak holds its place with a dash")
        #expect(ActivityText.memoryHelp(arrays).contains("does not measure its weights apart"))
        // Growth since the runner's last reading is drawn and counted as working memory, never as "outside MLX".
        let growing = ActivityFeed.Memory(source: "mlx", totalMB: 12_000, weightsMB: 8_400, workingMB: 0, cacheMB: 100, unsplitMB: 3_000, peakMB: 12_500)
        #expect(ActivityText.memoryBar(growing).map(\.mb) == [8_400, 3_000, 100, 500])
        #expect(ActivityText.memoryLegend(growing).map(\.mb) == [8_400, 3_100, 12_500])
        #expect(ActivityText.memoryHelp(growing).contains("grew during this job"))
        let process = ActivityFeed.Memory(source: "process", totalMB: 2_150, peakMB: 3_400)
        #expect(ActivityText.memoryBar(process).map(\.mb) == [2_150])
        #expect(ActivityText.memoryLegend(process).map(\.name) == ["In use", "Peak"])
        #expect(ActivityText.memoryHelp(process).contains("does not report MLX\u{2019}s counters"))
    }

    /// The breakdown arriving adds the card's memory line (a resize); its figures changing every second do not.
    @Test func memoryFiguresChangingNeverResizeThePanel() {
        let model = ActivityModel()
        var m = ActivityFeed.Model(id: "seedvr2-3b", kind: "restoration", state: "step", requests: [], queued: 0, batch: nil, memoryMB: 9_000)
        let without = model.layout(ActivityFeed(models: [m]))
        m.memory = .init(source: "mlx", totalMB: 9_000, weightsMB: 8_400, workingMB: 300, cacheMB: 0)
        let with = model.layout(ActivityFeed(models: [m]))
        #expect(with != without)
        m.memory?.workingMB = 21_000; m.memory?.peakMB = 37_000
        #expect(model.layout(ActivityFeed(models: [m])) == with)
    }

    /// At real size: the card with its memory line fits the panel's width, and is taller by the line alone.
    @Test func theCardDrawsItsMemoryLineAtTheMenuWidth() async {
        let model = ActivityModel()
        model.rows = [
            "seedvr2-3b": ModelRow(
                id: "seedvr2-3b", name: "SeedVR2 3B", kind: "restoration", precisions: ["fp16"], states: ["fp16": .init(state: .loaded)], selectedPrecision: "fp16")
        ]
        let plain = ActivityFeed.Model(id: "seedvr2-3b", kind: "restoration", state: "idle", requests: [], queued: 0, batch: nil, memoryMB: 9_000)
        var split = plain
        split.memory = .init(source: "mlx", totalMB: 9_000, weightsMB: 8_400, workingMB: 300, cacheMB: 40, peakMB: 37_000, mlxPeakMB: 36_500)
        func height(_ m: ActivityFeed.Model) -> CGFloat {
            model.preview(ActivityFeed(models: [m], machine: ActivityDesignRenders.machineIdle))
            let view = ActivityHostingView(rootView: ActivityPanel(model: model))
            view.setNaturalSize(NSSize(width: ActivityPanel.width, height: ActivityPanel.minHeight))
            return ActivityMenu.height(view)
        }
        let before = height(plain), after = height(split)
        #expect(after > before + 12 && after < before + 40, "\(before) → \(after)")
    }
}
