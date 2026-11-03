// SPDX-License-Identifier: MIT
import Foundation
import ServiceManagement

/// `Arsenal login on|off|status`: Launch at Login for the bundle this executable belongs to, without starting the app.
enum LoginCommand {
    static func run(_ arguments: [String], bundleIdentifier: String? = Bundle.main.bundleIdentifier) -> Int32 {
        guard arguments.count == 1, let action = arguments.first, ["on", "off", "status"].contains(action) else {
            FileHandle.standardError.write(Data("usage: Arsenal login on|off|status\n".utf8)); return 2
        }
        guard bundleIdentifier != nil else {
            FileHandle.standardError.write(Data("Launch at Login needs the installed Arsenal.app, not a bare binary\n".utf8)); return 1
        }
        let service = SMAppService.mainApp
        do {
            if action == "on", service.status != .enabled { try service.register() }
            if action == "off", service.status != .notRegistered { try service.unregister() } // approval pending too
        } catch {
            FileHandle.standardError.write(Data("Launch at Login: \(error.localizedDescription)\n".utf8)); return 1
        }
        print("Launch at Login: \(service.status == .enabled ? "on" : "off")")
        return 0
    }
}
