// Build-time icon renderer (not shipped): rasterizes the Arsenal cannon mark from its source SVGs, never a redrawing.
// Usage: swiftc icon.swift -o icon && icon <icon.svg> <glyph.svg> <out-dir> [<icon-16.svg> <icon-32.svg>]
// Writes <out-dir>/app-icon.png (1024², from icon.svg), optional app-icon-16.png / app-icon-32.png from the hand-tuned
// small-size sources, plus MenuBarTemplate.png and MenuBarTemplate@2x.png from glyph.svg, the monochrome menu-bar glyph,
// at 16 and 32 px high with the glyph's own aspect (22×16 for the cannon, so 22×16 and 44×32). The "Template" suffix makes AppKit treat the menu-bar image as
// a template (tinted by the menu bar).
// AppKit renders SVG natively, but rasterizes an SVG <mask> at the drawing's user-unit resolution (16 px for the glyph),
// which blurs the cut between barrel and wheel. Masked elements are therefore rendered here at full size: the mask's
// own content (a luminance image) and the element without its mask, combined pixel by pixel as SVG defines it.
import AppKit

func bitmap(_ width: Int, _ height: Int, _ draw: (CGFloat, CGFloat) -> Void) -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4,
        hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)!
    draw(CGFloat(width), CGFloat(height))
    NSGraphicsContext.restoreGraphicsState()
    return rep
}

/// One SVG document rendered into a bitmap of `width`×`height`.
func render(_ document: XMLDocument, width: Int, height: Int) -> NSBitmapImageRep {
    guard let image = NSImage(data: document.xmlData) else { fatalError("AppKit could not read the SVG") }
    return bitmap(width, height) { w, h in image.draw(in: NSRect(x: 0, y: 0, width: w, height: h)) }
}

/// Width over height of the document's viewBox (1 when it has none).
func aspect(_ document: XMLDocument) -> Double {
    guard let box = document.rootElement()?.attribute(forName: "viewBox")?.stringValue else { return 1 }
    let parts = box.split(whereSeparator: { $0 == " " || $0 == "," }).compactMap { Double($0) }
    return parts.count == 4 && parts[3] > 0 ? parts[2] / parts[3] : 1
}

func duplicate<T>(_ value: NSObject, as type: T.Type) -> T {
    guard let copy = value.copy() as? T else { fatalError("could not copy \(value)") }
    return copy
}

func elements(_ node: XMLNode) -> [XMLElement] {
    (node.children ?? []).compactMap { $0 as? XMLElement }.flatMap { [$0] + elements($0) }
}

/// The source with only `keep(element)` drawn: other drawable leaves are removed (definitions stay).
func filtered(_ source: XMLDocument, keep: (XMLElement) -> Bool, transform: (XMLElement) -> Void = { _ in }) -> XMLDocument {
    let copy = duplicate(source, as: XMLDocument.self)
    for element in elements(copy.rootElement()!) where element.name != "g" && element.name != "svg" {
        if element.parent?.name == "mask" || element.name == "mask" { continue }
        if keep(element) { transform(element) } else { element.detach() }
    }
    return copy
}

/// The SVG rasterized at `height` px high (width from its viewBox aspect), with every `mask="url(#id)"` applied at full resolution.
func rasterize(_ source: XMLDocument, height pixels: Int) -> NSBitmapImageRep {
    let width = Int((Double(pixels) * aspect(source)).rounded())
    let root = source.rootElement()!
    let masked = elements(root).filter { $0.attribute(forName: "mask") != nil }
    let base = render(filtered(source) { $0.attribute(forName: "mask") == nil }, width: width, height: pixels)
    for element in masked {
        let reference = element.attribute(forName: "mask")!.stringValue!
        guard let id = reference.range(of: #"(?<=url\(#)[^)]+"#, options: .regularExpression).map({ String(reference[$0]) }),
            let mask = elements(root).first(where: { $0.name == "mask" && $0.attribute(forName: "id")?.stringValue == id })
        else { fatalError("unsupported mask reference \(reference)") }
        let path = element.xPath ?? ""
        let shape = render(filtered(source, keep: { $0.xPath == path }, transform: { $0.removeAttribute(forName: "mask") }), width: width, height: pixels)
        // The mask's content as its own drawing (same viewBox), so its luminance is rasterized at full size too.
        let maskDocument = duplicate(source, as: XMLDocument.self)
        let maskRoot = maskDocument.rootElement()!
        for child in maskRoot.children ?? [] { child.detach() }
        for child in mask.children ?? [] { maskRoot.addChild(duplicate(child, as: XMLNode.self)) }
        let luminance = render(maskDocument, width: width, height: pixels)
        for y in 0..<pixels {
            for x in 0..<width {
                guard let colour = shape.colorAt(x: x, y: y), colour.alphaComponent > 0,
                    let m = luminance.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB)
                else { continue }
                let weight = (0.2125 * m.redComponent + 0.7154 * m.greenComponent + 0.0721 * m.blueComponent) * m.alphaComponent
                let a = colour.alphaComponent * weight
                guard a > 0, let under = base.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
                    let over = colour.usingColorSpace(.deviceRGB)
                else { continue }
                // Source-over of the masked shape onto what is already drawn.
                let outA = a + under.alphaComponent * (1 - a)
                func mix(_ o: CGFloat, _ u: CGFloat) -> CGFloat { outA == 0 ? 0 : (o * a + u * under.alphaComponent * (1 - a)) / outA }
                base.setColor(
                    NSColor(
                        deviceRed: mix(over.redComponent, under.redComponent), green: mix(over.greenComponent, under.greenComponent),
                        blue: mix(over.blueComponent, under.blueComponent), alpha: outA), atX: x, y: y)
            }
        }
    }
    return base
}

func write(_ rep: NSBitmapImageRep, _ url: URL) throws {
    try rep.representation(using: .png, properties: [:])!.write(to: url)
}

guard [4, 6].contains(CommandLine.arguments.count) else {
    fputs("usage: icon <icon.svg> <glyph.svg> <out-dir> [<icon-16.svg> <icon-32.svg>]\n", stderr); exit(2)
}
let icon = try XMLDocument(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
let glyph = try XMLDocument(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]))
let out = URL(fileURLWithPath: CommandLine.arguments[3])
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
try write(rasterize(icon, height: 1024), out.appendingPathComponent("app-icon.png"))
if CommandLine.arguments.count == 6 {
    for (index, pixels) in [(4, 16), (5, 32)] {
        let small = try XMLDocument(contentsOf: URL(fileURLWithPath: CommandLine.arguments[index]))
        try write(rasterize(small, height: pixels), out.appendingPathComponent("app-icon-\(pixels).png"))
    }
}
for (name, pixels) in [("MenuBarTemplate.png", 16), ("MenuBarTemplate@2x.png", 32)] {
    try write(rasterize(glyph, height: pixels), out.appendingPathComponent(name))
}
