// SPDX-License-Identifier: MIT
import AppKit
import Combine
import ArsenalCore
import SwiftUI

// The Arsenal palette: opaque colours instead of the system's translucent label greys, so
// text keeps its contrast on the darker panel. One source for the browser and the table; the precision tints, the
// flame, the hot-row gold and the switch keep their own colours.

struct ArsenalPalette: Equatable {
    var canvas: NSColor, sidebar: NSColor, raised: NSColor, raisedBorder: NSColor, hover: NSColor, selFill: NSColor, hairline: NSColor, field: NSColor
    var t1: NSColor, t2: NSColor, t3: NSColor, t4: NSColor
    var accent: NSColor, accentText: NSColor, flag: NSColor, amber: NSColor, green: NSColor, red: NSColor
    /// The loaded row's wash.
    var hotRow: NSColor
    /// Get more models: the light-green wash of a model already on the table (blue stays the loaded row's).
    var addedRow = ArsenalPalette.rgb(0x6BD185, 0.24)
    var translucent = false
    var dark = true

    static func rgb(_ hex: UInt32, _ alpha: CGFloat = 1) -> NSColor {
        NSColor(srgbRed: CGFloat((hex >> 16) & 255) / 255, green: CGFloat((hex >> 8) & 255) / 255, blue: CGFloat(hex & 255) / 255, alpha: alpha)
    }
    static let dark = ArsenalPalette(
        canvas: rgb(0x111215), sidebar: rgb(0x16171B), raised: rgb(0x1B1D22), raisedBorder: rgb(0x2C2F36), hover: rgb(0x1A1C20), selFill: rgb(0x24272E),
        hairline: rgb(0x24262B), field: rgb(0x1F2126),
        t1: rgb(0xF1F2F4), t2: rgb(0xA3A8B1), t3: rgb(0x6E747E), t4: rgb(0x454A53),
        accent: rgb(0x3D7BF5), accentText: rgb(0x8EB4FF), flag: rgb(0x8EAFF6), amber: rgb(0xF2BD66), green: rgb(0x6BD185), red: rgb(0xFF857A),
        hotRow: NSColor.selectedContentBackgroundColor.withAlphaComponent(0.6))
    static let light = ArsenalPalette(
        canvas: rgb(0xFBFBFC), sidebar: rgb(0xF1F2F4), raised: rgb(0xFFFFFF), raisedBorder: rgb(0xD9DCE1), hover: rgb(0xF0F1F3), selFill: rgb(0xE2E5EA),
        hairline: rgb(0xE3E5E9), field: rgb(0xE6E8EB),
        t1: rgb(0x121317), t2: rgb(0x4B505A), t3: rgb(0x7D828C), t4: rgb(0xB6BAC1),
        accent: rgb(0x2F6BE8), accentText: rgb(0x245DD6), flag: rgb(0x3567D6), amber: rgb(0xA66300), green: rgb(0x1F8A40), red: rgb(0xC8372D),
        hotRow: rgb(0x2F6BE8), addedRow: rgb(0x34C759, 0.16), dark: false)
    /// Dark tokens over a vibrant window: the surfaces become tints, text and accents stay.
    static let translucent: ArsenalPalette = {
        var p = dark
        p.canvas = rgb(0x0D0E11, 0.80); p.sidebar = rgb(0x0D0E11, 0.72); p.raised = rgb(0x22252B, 0.86)
        // Secondary and quiet text one step lighter: what shows through can be bright.
        p.t2 = rgb(0xB4B9C1); p.t3 = rgb(0x8A9099)
        p.hover = NSColor.white.withAlphaComponent(0.04); p.selFill = NSColor.white.withAlphaComponent(0.09); p.field = NSColor.white.withAlphaComponent(0.07)
        p.hairline = NSColor.white.withAlphaComponent(0.08); p.raisedBorder = NSColor.white.withAlphaComponent(0.10)
        p.translucent = true
        return p
    }()
    static func of(_ a: RenderAppearance) -> ArsenalPalette { a == .light ? light : a == .translucent ? translucent : dark }

    // Vella's table accents (Projects/Vella Sources/Vella/ModelsTable/TierControl.swift, ModelTable.swift), the same in
    // both appearances: the precision control's blue tint, the loaded row's gold, its white text, and the deltas' lighter
    // green and red that stay legible on the loaded row's blue.
    /// Vella's light blue (HSL 221° 85 % 76 %): Get more models' precision pill on a model that is not loaded.
    static let tint = rgb(0x8EAFF6)
    /// The chosen precision (TierControl.gold, Vella 2.1.0's gold): the table's chosen cell, the pending status line,
    /// Reload, and Get more models' loaded precision.
    static let gold = TierControl.gold
    /// Labels on the tint and the gold.
    static let ink = TierControl.ink
    /// Text on the loaded row.
    static let hotText = NSColor.white
    static let hotGreen = rgb(0x9EF5AD), hotRed = rgb(0xFFBDB3)
    /// The bolt glyph: Vella's lemon yellow on dark, its amber on the light menu (TierControl.boltColor).
    var bolt: NSColor { dark ? TierControl.boltTint : TierControl.boltTintLight }
    /// Text colours on a loaded row's blue (Get more models' loaded rows): white steps, the table's light green, red and
    /// gold for its tones.
    var onLoadedRow: ArsenalPalette {
        var p = self
        p.t1 = Self.hotText; p.t2 = Self.hotText.withAlphaComponent(0.85); p.t3 = Self.hotText.withAlphaComponent(0.72)
        p.green = Self.hotGreen; p.red = Self.hotRed; p.amber = Self.gold
        return p
    }

    /// A neutral overlay for fills and strokes drawn over a surface: white on dark, black on light.
    func overlay(_ alpha: CGFloat) -> NSColor { dark ? NSColor.white.withAlphaComponent(alpha) : NSColor.black.withAlphaComponent(alpha * 0.6) }
    /// For content drawn straight on the menu's own material (the Models table, the Activity panel): the menu supplies
    /// the background, so surfaces, lines and fills become overlays that read on whatever the material shows, and the
    /// quieter text steps are lifted for the lighter dark material. `t4` (unavailable: greyed in place, never hidden) is
    /// the label colour at about 40 %, so it keeps its contrast over whatever the translucent material shows. `canvas`
    /// stays the dark ink drawn on accent fills.
    static func menu(dark: Bool) -> ArsenalPalette {
        var p = dark ? Self.dark : Self.light
        if dark {
            p.t2 = rgb(0xB4B9C1); p.t3 = rgb(0x8E949D)
        } else {
            p.t3 = rgb(0x6C717B)
        }
        p.t4 = p.t1.withAlphaComponent(0.42)
        // Nothing in the menu is a separate opaque slab.
        p.raised = .clear
        p.hairline = p.overlay(0.10); p.raisedBorder = p.overlay(0.15); p.field = p.overlay(0.08)
        p.hover = p.overlay(0.05); p.selFill = p.overlay(0.10)
        return p
    }
    /// The menu palette now (the table's AppKit cells read it at draw time).
    @MainActor static var onMenu: ArsenalPalette { Appearance.shared.menuPalette }
    /// The windows' palette now (the tools' AppKit views read it at draw time).
    @MainActor static var onWindow: ArsenalPalette { Appearance.shared.palette }
}

/// System appearance drives every window and menu; there is no app appearance preference.
enum RenderAppearance: String { case dark, light, translucent }
@MainActor final class Appearance: ObservableObject {
    static let shared = Appearance()
    @Published private(set) var mode: RenderAppearance = .dark
    var palette: ArsenalPalette { ArsenalPalette.of(mode) }
    /// Content hosted in a menu draws on the menu's native material (see `ArsenalPalette.menu`).
    var menuPalette: ArsenalPalette { mode == .light ? Self.menuLight : Self.menuDark }
    private static let menuDark = ArsenalPalette.menu(dark: true), menuLight = ArsenalPalette.menu(dark: false)
    var nsAppearance: NSAppearance? { NSAppearance(named: mode == .light ? .aqua : .darkAqua) }
    var colorScheme: ColorScheme { mode == .light ? .light : .dark }
    private var observation: NSKeyValueObservation?
    private init() {
        followSystem()
        observation = NSApplication.shared.observe(\.effectiveAppearance, options: [.new]) { _, _ in
            Task { @MainActor in Appearance.shared.followSystem() }
        }
    }
    func followSystem() {
        mode = NSApplication.shared.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .aqua ? .light : .dark
    }
    #if DEBUG
        /// Offscreen render fixtures only; no persisted setting or production control.
        func set(_ mode: RenderAppearance) { self.mode = mode }
    #endif
}

extension NSColor { var swiftUI: Color { Color(nsColor: self) } }
