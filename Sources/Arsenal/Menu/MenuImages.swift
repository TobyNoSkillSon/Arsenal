// SPDX-License-Identifier: MIT
import AppKit

extension NSMenuItem {
    func showMenuImage() {
        if #available(macOS 27.0, *) { setValue(1, forKey: "preferredImageVisibility") }
    }
}
