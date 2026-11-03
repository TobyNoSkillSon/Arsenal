// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import AudioWorker

/// Bug (2 Nov 2026 review): Nemotron 3 Diarization sessions never expired, so two crashed clients blocked every later
/// create until the model was unloaded. Abandoned sessions now give up their slot after ten idle minutes, as Sortformer
/// v2's do. A one-layer random network; creating a session runs no inference.
@Test func nemotronAbandonedSessionsFreeTheirSlot() throws {
    useTestMetallib()
    let adapter = NemotronAdapter()
    let backend = ArsenalNemotronBackend(Nemotron3MLXNetwork(layerCount: 1))
    adapter.backend = backend
    adapter.diarizer = Nemotron3Diarizer(backend: backend)
    var clock = 1_000.0
    adapter.now = { clock }
    try expectAbandonedSessionsFreeTheirSlot(adapter, advance: { clock += $0 })
}
