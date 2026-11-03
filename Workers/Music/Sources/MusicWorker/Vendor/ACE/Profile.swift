//
//  Profile.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  Stage timings on stderr when ACESTEP_PROFILE is set; silent otherwise.
//

import Foundation

enum Profile {
    static let enabled = ProcessInfo.processInfo.environment["ACESTEP_PROFILE"] != nil

    static func log(_ line: String) {
        guard enabled else { return }
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }
}
