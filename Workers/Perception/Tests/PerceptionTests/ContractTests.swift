import Foundation
import Testing
import PerceptionWorker
import MLXRFDETR
import MLXTIPS
import SAM31
import MLXLocateAnything

@Test func registrationsAreComplete() {
    let models = Set(PerceptionWorker.registrations.flatMap { $0.modelTypes })
    #expect(models == ["tipsv2", "rf_detr", "locateanything", "sam3.1_video"])
    #expect(PerceptionWorker.registrations.allSatisfy { $0.conversion != nil })
}
@Test func cocoHasExactlyEightyNamedClasses() {
    #expect(NativeRFDETR.cocoLabels.values.compactMap(\.string).filter { $0 != "N/A" }.count == 80)
    #expect(NativeRFDETR.cocoLabels["1"]?.string == "person")
    #expect(NativeRFDETR.cocoLabels["90"]?.string == "toothbrush")
}
@Test func modelCapabilitiesStayNarrow() {
    #expect(NativeTIPS().operations == ["embed", "detect", "segment"])
    #expect(NativeRFDETR().operations == ["detect"])
    #expect(NativeLocateAnything().operations == ["detect", "ground_screen"])
    #expect(NativeSAM31().operations == ["segment", "segment_session", "video_session", "track_video"])
}

@Test func locateBoxesAndPointsKeepOriginalPixelScale() throws {
    let config = try JSONDecoder().decode(LocateAnythingConfiguration.self, from: Data("{\"text_config\":{},\"vision_config\":{},\"eos_token_id\":151645}".utf8))
    let coordinates = [250, 400, 750, 900].map { config.coordStartTokenId + $0 }
    let boxes = try LocateAnythingBoxes.parse([config.boxStartTokenId] + coordinates + [config.boxEndTokenId], config: config, imageWidth: 800, imageHeight: 600) { _ in "button" }
    #expect(boxes.count == 1)
    #expect(boxes[0].box == [0.25, 0.4, 0.75, 0.9])
    #expect(boxes[0].boxPixels == [200, 240, 600, 540])
    let pointTokens = [config.boxStartTokenId] + Array(coordinates.prefix(2)) + [config.boxEndTokenId]
    let points = try LocateAnythingBoxes.parse(pointTokens, config: config, imageWidth: 800, imageHeight: 600) { _ in "button" }
    #expect(points.count == 1 && points[0].isPoint)
    #expect(points[0].boxPixels == [200, 240])
}
