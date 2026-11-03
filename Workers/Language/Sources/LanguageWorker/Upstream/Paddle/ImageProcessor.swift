import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import MLX

enum ProcessingMode: String, CaseIterable, Sendable {
    case base
    case dynamic

    var imageSize: Int {
        switch self {
        case .base: return 448
        case .dynamic: return 448
        }
    }

    var dynamicResolution: Bool {
        switch self {
        case .dynamic: return true
        default: return false
        }
    }
}

struct ProcessedImages {
    let pixelValues: MLXArray
    let width: Int
    let height: Int
    let numImageTokens: Int
}

struct BatchProcessedImages {
    let items: [ProcessedImages]

    var batchSize: Int { items.count }

    var maxNumImageTokens: Int {
        items.map { $0.numImageTokens }.max() ?? 0
    }

    var canBatchVision: Bool {
        guard let first = items.first else { return false }
        return items.allSatisfy { $0.numImageTokens == first.numImageTokens }
    }

    func stackedPixelValues() -> MLXArray {
        let views = items.map { $0.pixelValues.squeezed(axis: 0) }
        return MLX.stacked(views, axis: 0)
    }
}

struct PaddleOCRVLImageProcessor {
    static let imageMean: (Float, Float, Float) = (0.5, 0.5, 0.5)
    static let imageStd: (Float, Float, Float) = (0.5, 0.5, 0.5)

    let imageSize: Int
    let dynamicResolution: Bool
    let patchSize: Int
    let minPixels: Int
    let maxPixels: Int

    private let context = CIContext()

    init(
        imageSize: Int = 448,
        dynamicResolution: Bool = false,
        patchSize: Int = 14,
        minPixels: Int = 112896,
        maxPixels: Int = 14 * 14 * 4 * 1280
    ) {
        self.imageSize = imageSize
        self.dynamicResolution = dynamicResolution
        self.patchSize = patchSize
        self.minPixels = minPixels
        self.maxPixels = maxPixels
    }

    init(mode: ProcessingMode) {
        self.imageSize = mode.imageSize
        self.dynamicResolution = mode.dynamicResolution
        self.patchSize = 14
        self.minPixels = 112896
        self.maxPixels = 14 * 14 * 4 * 1280
    }

    func process(imageAt path: String) throws -> ProcessedImages {
        let url = URL(fileURLWithPath: path)
        return try process(imageAt: url)
    }

    func process(imageAt url: URL) throws -> ProcessedImages {
        guard let ciImage = CIImage(contentsOf: url) else {
            throw PaddleOCRVLError.imageLoadFailed(url.path)
        }
        return process(ciImage)
    }

    func process(_ image: CIImage) -> ProcessedImages {
        if dynamicResolution {
            return processDynamicResolution(image)
        } else {
            return processFixedResolution(image)
        }
    }

    private func processFixedResolution(_ image: CIImage) -> ProcessedImages {
        let targetSize = CGSize(width: CGFloat(imageSize), height: CGFloat(imageSize))
        let resizedImage = resizeWithPadding(image, to: targetSize)
        let pixelValues = normalizeAndConvert(resizedImage)

        let numPatches = imageSize / patchSize
        let numImageTokens = numPatches * numPatches

        return ProcessedImages(
            pixelValues: pixelValues,
            width: imageSize,
            height: imageSize,
            numImageTokens: numImageTokens / 4
        )
    }

    private func processDynamicResolution(_ image: CIImage) -> ProcessedImages {
        let originalWidth = Int(image.extent.width)
        let originalHeight = Int(image.extent.height)

        let (targetWidth, targetHeight) = smartResize(
            width: originalWidth,
            height: originalHeight
        )

        let targetSize = CGSize(width: CGFloat(targetWidth), height: CGFloat(targetHeight))
        let resizedImage = resizeDirectly(image, to: targetSize)
        let pixelValues = normalizeAndConvert(resizedImage)

        let numPatchesW = targetWidth / patchSize
        let numPatchesH = targetHeight / patchSize
        let numImageTokens = numPatchesW * numPatchesH

        return ProcessedImages(
            pixelValues: pixelValues,
            width: targetWidth,
            height: targetHeight,
            numImageTokens: numImageTokens / 4
        )
    }

    private func smartResize(width: Int, height: Int) -> (Int, Int) {
        let factor = patchSize * 2
        var width = Double(width), height = Double(height)
        if height < Double(factor) { width = (width * Double(factor) / height).rounded(.toNearestOrEven); height = Double(factor) }
        if width < Double(factor) { height = (height * Double(factor) / width).rounded(.toNearestOrEven); width = Double(factor) }
        let unit = Double(factor)
        var h = (height / unit).rounded(.toNearestOrEven) * unit, w = (width / unit).rounded(.toNearestOrEven) * unit
        if h * w > Double(maxPixels) {
            let beta = sqrt(height * width / Double(maxPixels))
            h = floor(height / beta / unit) * unit; w = floor(width / beta / unit) * unit
        } else if h * w < Double(minPixels) {
            let beta = sqrt(Double(minPixels) / (height * width))
            h = ceil(height * beta / unit) * unit; w = ceil(width * beta / unit) * unit
        }
        return (max(factor, Int(w)), max(factor, Int(h)))
    }

    private func resizeWithPadding(_ image: CIImage, to size: CGSize) -> CIImage {
        let originalWidth = image.extent.width
        let originalHeight = image.extent.height

        let scaleX = size.width / originalWidth
        let scaleY = size.height / originalHeight
        let scale = min(scaleX, scaleY)

        let scaledWidth = originalWidth * scale
        let scaledHeight = originalHeight * scale

        let scaleTransform = CGAffineTransform(scaleX: scale, y: scale)
        var scaledImage = image.transformed(by: scaleTransform)

        let padX = (size.width - scaledWidth) / 2
        let padY = (size.height - scaledHeight) / 2

        let translateTransform = CGAffineTransform(translationX: padX, y: padY)
        scaledImage = scaledImage.transformed(by: translateTransform)

        let fillColor = CIColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        let backgroundRect = CGRect(origin: .zero, size: size)
        let background = CIImage(color: fillColor).cropped(to: backgroundRect)

        let composited = scaledImage.composited(over: background)
        return composited.cropped(to: backgroundRect)
    }

    private func resizeDirectly(_ image: CIImage, to size: CGSize) -> CIImage {
        // PIL BICUBIC (Keys a=-0.5) with clamped edges, as the maker's processor resizes.
        DocumentRegions.resize(image, width: Int(size.width), height: Int(size.height))
    }

    private func normalizeAndConvert(_ image: CIImage) -> MLXArray {
        let rawArray = asMLXArrayRaw(image)
        let mean = MLXArray([Self.imageMean.0, Self.imageMean.1, Self.imageMean.2])
        let std = MLXArray([Self.imageStd.0, Self.imageStd.1, Self.imageStd.2])
        let normalized = (rawArray - mean) / std
        return normalized.asType(.float16)
    }

    private func asMLXArrayRaw(_ image: CIImage) -> MLXArray {
        let size = image.extent.size
        let w = Int(size.width.rounded())
        let h = Int(size.height.rounded())

        let format = CIFormat.RGBA8
        let bytesPerPixel = 4
        let bytesPerRow = w * bytesPerPixel

        var data = Data(count: w * h * bytesPerPixel)
        data.withUnsafeMutableBytes { ptr in
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
            context.render(
                image,
                toBitmap: ptr.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: image.extent,
                format: format,
                colorSpace: colorSpace
            )
            context.clearCaches()
        }

        let uint8Array = MLXArray(data, [h, w, 4], type: UInt8.self)
        var array = uint8Array.asType(.float32) / 255.0
        array = array[0..., 0..., ..<3]
        array = array.reshaped(1, h, w, 3)

        return array
    }

    func processBatch(imagesAt paths: [String]) throws -> BatchProcessedImages {
        let items = try paths.map { try process(imageAt: $0) }
        return BatchProcessedImages(items: items)
    }

    func processBatch(_ images: [CIImage]) -> BatchProcessedImages {
        let items = images.map { process($0) }
        return BatchProcessedImages(items: items)
    }
}

enum PaddleOCRVLError: LocalizedError {
    case imageLoadFailed(String)
    case modelLoadFailed(String)
    case tokenizerLoadFailed(String)
    case imageRequired
    case configurationError(String)
    case generationError(String)

    var errorDescription: String? {
        switch self {
        case .imageLoadFailed(let path):
            return "Failed to load image at: \(path)"
        case .modelLoadFailed(let message):
            return "Failed to load model: \(message)"
        case .tokenizerLoadFailed(let message):
            return "Failed to load tokenizer: \(message)"
        case .imageRequired:
            return "Image is required for OCR"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .generationError(let message):
            return "Generation error: \(message)"
        }
    }
}
