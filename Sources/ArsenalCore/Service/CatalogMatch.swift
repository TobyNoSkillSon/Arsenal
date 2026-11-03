// SPDX-License-Identifier: MIT
import Foundation

/// Shared variant selection and decimal-unit memory boundary for every catalog surface.
public enum CatalogMatch {
    public static func orderedDtypes(_ entry: CatalogEntry) -> [String] {
        func bits(_ dtype: String) -> Double {
            entry.formats.first { $0.id == dtype }?.bitsPerWeight ?? (dtype == "int4" ? 4.5 : dtype == "int8" ? 8.5 : 16)
        }
        return entry.dtypes.enumerated().sorted { a, b in
            let left = bits(a.element), right = bits(b.element)
            return left == right ? a.offset < b.offset : left > right
        }.map(\.element)
    }
    public static func matchingDtype(_ entry: CatalogEntry, satisfies: (String) -> Bool) -> String? {
        let selected = entry.row.selectedPrecision
        return ([selected] + orderedDtypes(entry).filter { $0 != selected }).first { entry.dtypes.contains($0) && satisfies($0) }
    }
    public static func matchingDtype(_ entry: CatalogEntry, query: CatalogQuery) -> String? {
        matchingDtype(entry) { dtype in
            guard query.dtypes.isEmpty || query.dtypes.contains(dtype) else { return false }
            let bytes = entry.sizesByDtype[dtype], ram = entry.ramMBByDtype[dtype]
            if let min = query.minSizeBytes, bytes.map({ $0 >= min }) != true { return false }
            if let max = query.maxSizeBytes, bytes.map({ $0 <= max }) != true { return false }
            if let min = query.minRAMMB, ram.map({ $0 >= min }) != true { return false }
            if let max = query.maxRAMMB, ram.map({ $0 <= max }) != true { return false }
            if let fits = query.fitsRAM, entry.fitsRAMByDtype[dtype] != fits { return false }
            return true
        }
    }
    /// Fits under this Mac's line (`MacMemoryGuard`: 100/128 of its RAM), the line Load admission refuses past, so a
    /// model Fits shows is one this Mac can load with nothing else running (Load admission also counts what macOS and
    /// open apps hold at that moment). Any Mac's RAM, never a fixed 128 GB.
    public static func fits(ramMB: Double, physicalBytes: UInt64) -> Bool {
        ramMB.isFinite && ramMB >= 0 && ramMB + MemoryProbe.headroomMB <= MacMemoryGuard.safeInUseMB(totalMB: Double(physicalBytes) / 1e6)
    }
}
