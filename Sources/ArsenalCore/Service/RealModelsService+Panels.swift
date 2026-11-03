// SPDX-License-Identifier: MIT
import Foundation

extension RealModelsService {
    /// The model's settings panel, kept beside its folder metadata when its folder was read (signed bytes or the local
    /// snapshot, validated against its declarations).
    public func settingsPanel(id: String) async -> SettingsPanel? { SettingsPanel(json: metadata[id]?[SettingsPanel.metadataKey]) }
}
