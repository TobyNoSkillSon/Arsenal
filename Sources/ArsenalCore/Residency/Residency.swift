// SPDX-License-Identifier: MIT
import Foundation

public enum ResidencyClass: String, Codable, Sendable { case manual, onDemand = "on_demand" }
public enum KeepHot {
    public static let choices = [5, 15, 30, 60, 0]
    public static func title(_ minutes: Int) -> String { minutes == 0 ? "Always" : "\(minutes) min idle" }
}
public struct ResidencySettings: Codable, Sendable, Equatable {
    public var manualIdleMinutes: Int
    public var onDemandIdleMinutes: Int
    public var allowSwap: Bool
    public init(manualIdleMinutes: Int = 0, onDemandIdleMinutes: Int = 15, allowSwap: Bool = false) {
        self.manualIdleMinutes = manualIdleMinutes; self.onDemandIdleMinutes = onDemandIdleMinutes; self.allowSwap = allowSwap
    }
    enum CodingKeys: String, CodingKey { case manualIdleMinutes, onDemandIdleMinutes, allowSwap }
    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            manualIdleMinutes: try values.decodeIfPresent(Int.self, forKey: .manualIdleMinutes) ?? 0,
            onDemandIdleMinutes: try values.decodeIfPresent(Int.self, forKey: .onDemandIdleMinutes) ?? 15,
            allowSwap: try values.decodeIfPresent(Bool.self, forKey: .allowSwap) ?? false)
    }
    public func idleMinutes(_ residency: ResidencyClass) -> Int { residency == .manual ? manualIdleMinutes : onDemandIdleMinutes }
}
public struct ResidentModel: Codable, Sendable {
    public var id: String; public var residency: ResidencyClass; public var lastUsed: Double; public var memoryMB: Double; public var busy: Int; public var keepHot: Bool
    public init(id: String, residency: ResidencyClass, lastUsed: Double = Date().timeIntervalSince1970, memoryMB: Double = 0, busy: Int = 0, keepHot: Bool = false) {
        self.id = id; self.residency = residency; self.lastUsed = lastUsed; self.memoryMB = memoryMB; self.busy = busy; self.keepHot = keepHot
    }
}
public enum AdmissionDecision: Equatable, Sendable { case admit(evict: [String]), refuse(needMB: Double, freeMB: Double) }
public enum ResidencyPolicy {
    public static func evictionOrder(_ models: [ResidentModel]) -> [ResidentModel] {
        models.sorted { a, b in a.residency == b.residency ? a.lastUsed < b.lastUsed : a.residency == .onDemand }
    }
    public static func admission(id: String, estimateMB: Double, availableMB: Double, residents: [ResidentModel], allowSwap: Bool, evictOnDemand: Bool = true) -> AdmissionDecision
    {
        let need = estimateMB + MemoryProbe.headroomMB
        if allowSwap || availableMB >= need { return .admit(evict: []) }
        var free = availableMB; var victims: [String] = []
        for r in evictionOrder(residents) where evictOnDemand && r.id != id && r.busy == 0 && r.residency == .onDemand && !r.keepHot {
            free += r.memoryMB; victims.append(r.id); if free >= need { return .admit(evict: victims) }
        }
        return .refuse(needMB: need, freeMB: max(0, availableMB))
    }
    public static func deadline(_ model: ResidentModel, settings: ResidencySettings, minuteSeconds: Double = 60) -> Double? {
        let minutes = settings.idleMinutes(model.residency)
        guard !model.keepHot, model.busy == 0, minutes > 0 else { return nil }
        return model.lastUsed + Double(minutes) * minuteSeconds
    }
    public static func pressureVictims(_ residents: [ResidentModel], order: [String]) -> [String] {
        let keep = order.first { id in residents.contains { $0.id == id && $0.residency == .manual } } ?? order.first
        return evictionOrder(residents).filter { $0.id != keep && $0.busy == 0 && $0.residency == .onDemand && !$0.keepHot }.map(\.id)
    }
}
