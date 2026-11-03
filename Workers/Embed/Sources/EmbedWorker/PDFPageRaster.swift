// SPDX-License-Identifier: MIT
import CoreGraphics

/// A PDF page as a viewer shows it: its crop box, turned by its /Rotate, filling a bitmap of up to 2× its point size
/// and at most 2,048 px on the long side. EmbeddingGemma 2 and LCO-Embedding both render pages through this.
enum PDFPageRaster {
    /// The visible page in points: the crop box, width and height swapped for a quarter turn.
    static func shownSize(_ page: CGPDFPage) -> CGSize {
        let crop = page.getBoxRect(.cropBox)
        return page.rotationAngle % 180 != 0 ? CGSize(width: crop.height, height: crop.width) : crop.size
    }

    static func pixelSize(_ shown: CGSize) -> (width: Int, height: Int) {
        let scale = min(2.0, 2048 / max(shown.width, shown.height))
        return (max(1, Int(shown.width * scale)), max(1, Int(shown.height * scale)))
    }

    /// Fills the context white, then draws the page across all `width`×`height` pixels. `getDrawingTransform` only
    /// ever scales down, so it is taken for a page-sized rect and the context is scaled up to the bitmap first.
    static func draw(_ page: CGPDFPage, in context: CGContext, width: Int, height: Int) {
        let shown = shownSize(page)
        context.setFillColor(CGColor(gray: 1, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.scaleBy(x: CGFloat(width) / shown.width, y: CGFloat(height) / shown.height)
        context.concatenate(page.getDrawingTransform(.cropBox, rect: CGRect(origin: .zero, size: shown), rotate: 0, preserveAspectRatio: true))
        context.drawPDFPage(page)
    }
}
