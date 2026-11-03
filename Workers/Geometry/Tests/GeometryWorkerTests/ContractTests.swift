import Testing
import IntegrationKit
@testable import GeometryWorker

@Test func registrySeparatesArchitectures() {
    let registrations = GeometryWorker.registrations
    #expect(registrations.count == 5)
    #expect(Set(registrations.flatMap(\.modelTypes)) == ["moge3", "da3small", "da3-small", "da3mono", "da3mono-large", "trellis2", "Trellis2ImageTo3DPipeline", "idol"])
    for registration in registrations {
        #expect(!registration.make().operations.isEmpty)
        #expect(registration.conversion != nil)
    }
}

@Test func sourceLayoutsAndSkyBranch() {
    #expect(GeometryWeights.mappedName("model.head.scratch.sky_output_conv2.0.weight", moge: false) == "head.sky_conv2_0.weight")
    #expect(GeometryWeights.mappedName("model.head.scratch.sky_output_conv2.2.bias", moge: false) == "head.sky_conv2_1.bias")
    #expect(GeometryWeights.mappedName("model.cam_dec.fc.weight", moge: false) == nil)
    #expect(GeometryWeights.mappedName("neck.resamplers.0.1.weight", moge: true) == "neck.resamplers.0.layers.1.weight")
    #expect(GeometryWeights.permutation("head.resize_layer0.weight", shape: [256, 128, 4, 4], moge: false) == [1, 2, 3, 0])
    #expect(GeometryWeights.permutation("head.projects.0.weight", shape: [256, 128, 1, 1], moge: false) == [0, 2, 3, 1])
    #expect(GeometryWeights.permutation("neck.input_blocks.0.weight", shape: [256, 1, 1, 128], moge: true) == nil)
}

@Test func eligibilityExcludesNonMatricesAndUnalignedInputs() {
    #expect(GeometryWeights.eligible("encoder.blocks.0.attn.qkv.weight", [192, 64]))
    #expect(!GeometryWeights.eligible("encoder.blocks.0.attn.qkv.bias", [192]))
    #expect(!GeometryWeights.eligible("head.conv.weight", [64, 3, 3, 64]))
    #expect(!GeometryWeights.eligible("head.linear.weight", [64, 66]))
    #expect(!GeometryWeights.eligible("backbone.mask_token", [1, 64]))
}

@Test func folderArchitectureDispatchesOriginalDA3Configs() throws {
    for architecture in ["da3small", "da3mono"] {
        for checkpoint in [["model_name": "original-da3"], ["model_type": "vendor-original"]] {
            let registration = try NativeRegistration.resolve(
                checkpoint: checkpoint.mapValues { .string($0) }, architecture: architecture, registrations: GeometryWorker.registrations)
            #expect(registration.make().architecture == architecture)
        }
    }
}
