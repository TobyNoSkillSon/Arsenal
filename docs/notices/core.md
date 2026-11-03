# Core port provenance

MemoryProbe in Sources/ArsenalCore/Residency/MemoryProbe.swift is copied and adapted from Projects/Vella/Sources/VellaCore/Residency/Residency.swift at commit f4c5c34685829704f6492c49f087628d84838d5a (Vella; relicensed MIT by its author, 2 Oct 2026). ResidencyPolicy reimplements the same manual/on-demand LRU, idle windows and memory-headroom mechanism for arbitrary kinds; worker framing is original Arsenal code. No MLX dependency exists in the root package.
