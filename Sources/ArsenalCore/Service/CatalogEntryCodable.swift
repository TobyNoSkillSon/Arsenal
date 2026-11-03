// SPDX-License-Identifier: MIT
import Foundation

extension CatalogEntry {
    private enum CodingKeys: String, CodingKey {
        case matchingDtype, formats, ramMBByDtype, fitsRAMByDtype, row, description, links, runners, settingsSummary, licenceURL, hfRepo, derivedDtypes, sourceDtype, licenceName,
            licenceConditions, languages, tasks, goodFor, limits, modalitiesIn, modalitiesOut, suggested, popularity, sizesByDtype, minRAMMB, fitsRAM, added, installed, loaded
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let row = try c.decodeIfPresent(ModelRow.self, forKey: .row) ?? ModelRow(from: decoder)
        self.init(row: row)
        description = try c.decodeIfPresent(String.self, forKey: .description) ?? row.summary
        links = try c.decodeIfPresent([String: String].self, forKey: .links) ?? [:]
        runners = try c.decodeIfPresent([CatalogRunner].self, forKey: .runners) ?? []
        settingsSummary = try c.decodeIfPresent([SettingState].self, forKey: .settingsSummary) ?? []
        licenceURL = try c.decodeIfPresent(String.self, forKey: .licenceURL) ?? ""
        hfRepo = try c.decodeIfPresent(String.self, forKey: .hfRepo) ?? ""
        let derived = try c.decodeIfPresent([String].self, forKey: .derivedDtypes) ?? []
        derivedDtypes = derived
        sourceDtype = try c.decodeIfPresent(String.self, forKey: .sourceDtype) ?? row.precisions.first { !derived.contains($0) } ?? ""
        licenceName = try c.decodeIfPresent(String.self, forKey: .licenceName)
        licenceConditions = try c.decodeIfPresent(String.self, forKey: .licenceConditions)
        languages = try c.decodeIfPresent([String].self, forKey: .languages) ?? []
        tasks = try c.decodeIfPresent([String].self, forKey: .tasks) ?? []
        goodFor = try c.decodeIfPresent([String].self, forKey: .goodFor) ?? []; limits = try c.decodeIfPresent([String].self, forKey: .limits) ?? []
        modalitiesIn = try c.decodeIfPresent([String].self, forKey: .modalitiesIn) ?? []
        modalitiesOut = try c.decodeIfPresent([String].self, forKey: .modalitiesOut) ?? []
        suggested = try c.decodeIfPresent(Bool.self, forKey: .suggested) ?? false
        popularity = try c.decodeIfPresent(JSONObject.self, forKey: .popularity) ?? [:]
        sizesByDtype = try c.decodeIfPresent([String: Int64].self, forKey: .sizesByDtype) ?? row.downloadBytes
        minRAMMB = try c.decodeIfPresent(Double.self, forKey: .minRAMMB) ?? minRAMMB
        fitsRAM = try c.decodeIfPresent(Bool.self, forKey: .fitsRAM) ?? fitsRAM
        added = try c.decodeIfPresent(Bool.self, forKey: .added) ?? false
        installed = try c.decodeIfPresent(Bool.self, forKey: .installed) ?? installed
        matchingDtype = try c.decodeIfPresent(String.self, forKey: .matchingDtype)
        formats = try c.decodeIfPresent([ModelFormat].self, forKey: .formats) ?? []
        ramMBByDtype = try c.decodeIfPresent([String: Double].self, forKey: .ramMBByDtype) ?? ramMBByDtype
        fitsRAMByDtype = try c.decodeIfPresent([String: Bool].self, forKey: .fitsRAMByDtype) ?? fitsRAMByDtype
        loaded = try c.decodeIfPresent(Bool.self, forKey: .loaded) ?? loaded
    }
}
