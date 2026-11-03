// SPDX-License-Identifier: MIT
// Fixed descriptors from header-only inspection (ZIP directory + data.pkl metadata, read with a non-executing
// unpickler) of pyannote-community/speaker-diarization-community-1@8a527374977391da736e0daaef26855d949d9685.
// Offsets are in Float32 elements of the named storage entry; permutations map OIHW/OIW to MLX OHWI/OWI.
// ParamSincFB's window_ and n_ buffers are recomputed by C1Sinc and deliberately not read.
import Foundation
struct C1TensorDescriptor {
    let component: String; let source: String; let name: String; let entry: String; let offset: Int; let shape: [Int]; let permutation: [Int]?
}
let c1Descriptors: [C1TensorDescriptor] = [
    .init(
        component: "segmentation", source: "sincnet.wav_norm1d.weight", name: "segmentation.wav.weight", entry: "pytorch_model.updated/data/0", offset: 0, shape: [1],
        permutation: nil),
    .init(
        component: "segmentation", source: "sincnet.wav_norm1d.bias", name: "segmentation.wav.bias", entry: "pytorch_model.updated/data/1", offset: 0, shape: [1],
        permutation: nil),
    .init(
        component: "segmentation",
        source: "sincnet.conv1d.0.filterbank.low_hz_", name: "segmentation.sinc.low_hz_", entry: "pytorch_model.updated/data/2", offset: 0,
        shape: [40, 1],
        permutation: nil),
    .init(
        component: "segmentation",
        source: "sincnet.conv1d.0.filterbank.band_hz_", name: "segmentation.sinc.band_hz_", entry: "pytorch_model.updated/data/3", offset: 0,
        shape: [40, 1],
        permutation: nil),
    .init(
        component: "segmentation", source: "sincnet.conv1d.1.weight", name: "segmentation.conv.0.weight", entry: "pytorch_model.updated/data/6", offset: 0, shape: [60, 80, 5],
        permutation: [0, 2, 1]),
    .init(
        component: "segmentation", source: "sincnet.conv1d.1.bias", name: "segmentation.conv.0.bias", entry: "pytorch_model.updated/data/7", offset: 0, shape: [60],
        permutation: nil),
    .init(
        component: "segmentation", source: "sincnet.conv1d.2.weight", name: "segmentation.conv.1.weight", entry: "pytorch_model.updated/data/8", offset: 0, shape: [60, 60, 5],
        permutation: [0, 2, 1]),
    .init(
        component: "segmentation", source: "sincnet.conv1d.2.bias", name: "segmentation.conv.1.bias", entry: "pytorch_model.updated/data/9", offset: 0, shape: [60],
        permutation: nil),
    .init(
        component: "segmentation", source: "sincnet.norm1d.0.weight", name: "segmentation.norm.0.weight", entry: "pytorch_model.updated/data/10", offset: 0, shape: [80],
        permutation: nil),
    .init(
        component: "segmentation", source: "sincnet.norm1d.0.bias", name: "segmentation.norm.0.bias", entry: "pytorch_model.updated/data/11", offset: 0, shape: [80],
        permutation: nil),
    .init(
        component: "segmentation", source: "sincnet.norm1d.1.weight", name: "segmentation.norm.1.weight", entry: "pytorch_model.updated/data/12", offset: 0, shape: [60],
        permutation: nil),
    .init(
        component: "segmentation", source: "sincnet.norm1d.1.bias", name: "segmentation.norm.1.bias", entry: "pytorch_model.updated/data/13", offset: 0, shape: [60],
        permutation: nil),
    .init(
        component: "segmentation", source: "sincnet.norm1d.2.weight", name: "segmentation.norm.2.weight", entry: "pytorch_model.updated/data/14", offset: 0, shape: [60],
        permutation: nil),
    .init(
        component: "segmentation", source: "sincnet.norm1d.2.bias", name: "segmentation.norm.2.bias", entry: "pytorch_model.updated/data/15", offset: 0, shape: [60],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_ih_l0", name: "segmentation.forward.0.x_proj.weight", entry: "pytorch_model.updated/data/16", offset: 0, shape: [512, 60],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_hh_l0", name: "segmentation.forward.0.h_proj.weight", entry: "pytorch_model.updated/data/16", offset: 30720,
        shape: [512, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_ih_l0", name: "segmentation.forward.0.x_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1372160, shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_hh_l0", name: "segmentation.forward.0.h_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1372672, shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_ih_l0_reverse", name: "segmentation.backward.0.x_proj.weight", entry: "pytorch_model.updated/data/16", offset: 96256,
        shape: [512, 60],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_hh_l0_reverse", name: "segmentation.backward.0.h_proj.weight", entry: "pytorch_model.updated/data/16", offset: 126976,
        shape: [512, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_ih_l0_reverse", name: "segmentation.backward.0.x_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1373184,
        shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_hh_l0_reverse", name: "segmentation.backward.0.h_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1373696,
        shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_ih_l1", name: "segmentation.forward.1.x_proj.weight", entry: "pytorch_model.updated/data/16", offset: 192512,
        shape: [512, 256],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_hh_l1", name: "segmentation.forward.1.h_proj.weight", entry: "pytorch_model.updated/data/16", offset: 323584,
        shape: [512, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_ih_l1", name: "segmentation.forward.1.x_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1374208, shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_hh_l1", name: "segmentation.forward.1.h_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1374720, shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_ih_l1_reverse", name: "segmentation.backward.1.x_proj.weight", entry: "pytorch_model.updated/data/16", offset: 389120,
        shape: [512, 256],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_hh_l1_reverse", name: "segmentation.backward.1.h_proj.weight", entry: "pytorch_model.updated/data/16", offset: 520192,
        shape: [512, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_ih_l1_reverse", name: "segmentation.backward.1.x_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1375232,
        shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_hh_l1_reverse", name: "segmentation.backward.1.h_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1375744,
        shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_ih_l2", name: "segmentation.forward.2.x_proj.weight", entry: "pytorch_model.updated/data/16", offset: 585728,
        shape: [512, 256],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_hh_l2", name: "segmentation.forward.2.h_proj.weight", entry: "pytorch_model.updated/data/16", offset: 716800,
        shape: [512, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_ih_l2", name: "segmentation.forward.2.x_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1376256, shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_hh_l2", name: "segmentation.forward.2.h_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1376768, shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_ih_l2_reverse", name: "segmentation.backward.2.x_proj.weight", entry: "pytorch_model.updated/data/16", offset: 782336,
        shape: [512, 256],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_hh_l2_reverse", name: "segmentation.backward.2.h_proj.weight", entry: "pytorch_model.updated/data/16", offset: 913408,
        shape: [512, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_ih_l2_reverse", name: "segmentation.backward.2.x_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1377280,
        shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_hh_l2_reverse", name: "segmentation.backward.2.h_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1377792,
        shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_ih_l3", name: "segmentation.forward.3.x_proj.weight", entry: "pytorch_model.updated/data/16", offset: 978944,
        shape: [512, 256],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_hh_l3", name: "segmentation.forward.3.h_proj.weight", entry: "pytorch_model.updated/data/16", offset: 1110016,
        shape: [512, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_ih_l3", name: "segmentation.forward.3.x_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1378304, shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_hh_l3", name: "segmentation.forward.3.h_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1378816, shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_ih_l3_reverse", name: "segmentation.backward.3.x_proj.weight", entry: "pytorch_model.updated/data/16", offset: 1175552,
        shape: [512, 256],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.weight_hh_l3_reverse", name: "segmentation.backward.3.h_proj.weight", entry: "pytorch_model.updated/data/16", offset: 1306624,
        shape: [512, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_ih_l3_reverse", name: "segmentation.backward.3.x_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1379328,
        shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "lstm.bias_hh_l3_reverse", name: "segmentation.backward.3.h_proj.bias", entry: "pytorch_model.updated/data/16", offset: 1379840,
        shape: [512],
        permutation: nil),
    .init(
        component: "segmentation", source: "linear.0.weight", name: "segmentation.linear.0.weight", entry: "pytorch_model.updated/data/17", offset: 0, shape: [128, 256],
        permutation: nil),
    .init(
        component: "segmentation", source: "linear.0.bias", name: "segmentation.linear.0.bias", entry: "pytorch_model.updated/data/18", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "segmentation", source: "linear.1.weight", name: "segmentation.linear.1.weight", entry: "pytorch_model.updated/data/19", offset: 0, shape: [128, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "linear.1.bias", name: "segmentation.linear.1.bias", entry: "pytorch_model.updated/data/20", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "segmentation", source: "classifier.weight", name: "segmentation.classifier.weight", entry: "pytorch_model.updated/data/21", offset: 0, shape: [7, 128],
        permutation: nil),
    .init(
        component: "segmentation", source: "classifier.bias", name: "segmentation.classifier.bias", entry: "pytorch_model.updated/data/22", offset: 0, shape: [7],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.conv1.weight", name: "embedding.conv1.weight", entry: "pytorch_model.updated/data/0", offset: 0, shape: [32, 1, 3, 3],
        permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.bn1.weight", name: "embedding.bn1.weight", entry: "pytorch_model.updated/data/1", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.bn1.bias", name: "embedding.bn1.bias", entry: "pytorch_model.updated/data/2", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.bn1.running_mean", name: "embedding.bn1.running_mean", entry: "pytorch_model.updated/data/3", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.bn1.running_var", name: "embedding.bn1.running_var", entry: "pytorch_model.updated/data/4", offset: 0, shape: [32], permutation: nil
    ),
    .init(
        component: "embedding", source: "resnet.layer1.0.conv1.weight", name: "embedding.layer1.0.conv1.weight", entry: "pytorch_model.updated/data/6", offset: 0,
        shape: [32, 32, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer1.0.bn1.weight", name: "embedding.layer1.0.bn1.weight", entry: "pytorch_model.updated/data/7", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.0.bn1.bias", name: "embedding.layer1.0.bn1.bias", entry: "pytorch_model.updated/data/8", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.0.bn1.running_mean", name: "embedding.layer1.0.bn1.running_mean", entry: "pytorch_model.updated/data/9", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.0.bn1.running_var", name: "embedding.layer1.0.bn1.running_var", entry: "pytorch_model.updated/data/10", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.0.conv2.weight", name: "embedding.layer1.0.conv2.weight", entry: "pytorch_model.updated/data/12", offset: 0,
        shape: [32, 32, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer1.0.bn2.weight", name: "embedding.layer1.0.bn2.weight", entry: "pytorch_model.updated/data/13", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.0.bn2.bias", name: "embedding.layer1.0.bn2.bias", entry: "pytorch_model.updated/data/14", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.0.bn2.running_mean", name: "embedding.layer1.0.bn2.running_mean", entry: "pytorch_model.updated/data/15", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.0.bn2.running_var", name: "embedding.layer1.0.bn2.running_var", entry: "pytorch_model.updated/data/16", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.1.conv1.weight", name: "embedding.layer1.1.conv1.weight", entry: "pytorch_model.updated/data/18", offset: 0,
        shape: [32, 32, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer1.1.bn1.weight", name: "embedding.layer1.1.bn1.weight", entry: "pytorch_model.updated/data/19", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.1.bn1.bias", name: "embedding.layer1.1.bn1.bias", entry: "pytorch_model.updated/data/20", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.1.bn1.running_mean", name: "embedding.layer1.1.bn1.running_mean", entry: "pytorch_model.updated/data/21", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.1.bn1.running_var", name: "embedding.layer1.1.bn1.running_var", entry: "pytorch_model.updated/data/22", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.1.conv2.weight", name: "embedding.layer1.1.conv2.weight", entry: "pytorch_model.updated/data/24", offset: 0,
        shape: [32, 32, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer1.1.bn2.weight", name: "embedding.layer1.1.bn2.weight", entry: "pytorch_model.updated/data/25", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.1.bn2.bias", name: "embedding.layer1.1.bn2.bias", entry: "pytorch_model.updated/data/26", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.1.bn2.running_mean", name: "embedding.layer1.1.bn2.running_mean", entry: "pytorch_model.updated/data/27", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.1.bn2.running_var", name: "embedding.layer1.1.bn2.running_var", entry: "pytorch_model.updated/data/28", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.2.conv1.weight", name: "embedding.layer1.2.conv1.weight", entry: "pytorch_model.updated/data/30", offset: 0,
        shape: [32, 32, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer1.2.bn1.weight", name: "embedding.layer1.2.bn1.weight", entry: "pytorch_model.updated/data/31", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.2.bn1.bias", name: "embedding.layer1.2.bn1.bias", entry: "pytorch_model.updated/data/32", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.2.bn1.running_mean", name: "embedding.layer1.2.bn1.running_mean", entry: "pytorch_model.updated/data/33", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.2.bn1.running_var", name: "embedding.layer1.2.bn1.running_var", entry: "pytorch_model.updated/data/34", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.2.conv2.weight", name: "embedding.layer1.2.conv2.weight", entry: "pytorch_model.updated/data/36", offset: 0,
        shape: [32, 32, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer1.2.bn2.weight", name: "embedding.layer1.2.bn2.weight", entry: "pytorch_model.updated/data/37", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.2.bn2.bias", name: "embedding.layer1.2.bn2.bias", entry: "pytorch_model.updated/data/38", offset: 0, shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.2.bn2.running_mean", name: "embedding.layer1.2.bn2.running_mean", entry: "pytorch_model.updated/data/39", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer1.2.bn2.running_var", name: "embedding.layer1.2.bn2.running_var", entry: "pytorch_model.updated/data/40", offset: 0,
        shape: [32],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.conv1.weight", name: "embedding.layer2.0.conv1.weight", entry: "pytorch_model.updated/data/42", offset: 0,
        shape: [64, 32, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer2.0.bn1.weight", name: "embedding.layer2.0.bn1.weight", entry: "pytorch_model.updated/data/43", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.bn1.bias", name: "embedding.layer2.0.bn1.bias", entry: "pytorch_model.updated/data/44", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.bn1.running_mean", name: "embedding.layer2.0.bn1.running_mean", entry: "pytorch_model.updated/data/45", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.bn1.running_var", name: "embedding.layer2.0.bn1.running_var", entry: "pytorch_model.updated/data/46", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.conv2.weight", name: "embedding.layer2.0.conv2.weight", entry: "pytorch_model.updated/data/48", offset: 0,
        shape: [64, 64, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer2.0.bn2.weight", name: "embedding.layer2.0.bn2.weight", entry: "pytorch_model.updated/data/49", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.bn2.bias", name: "embedding.layer2.0.bn2.bias", entry: "pytorch_model.updated/data/50", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.bn2.running_mean", name: "embedding.layer2.0.bn2.running_mean", entry: "pytorch_model.updated/data/51", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.bn2.running_var", name: "embedding.layer2.0.bn2.running_var", entry: "pytorch_model.updated/data/52", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.shortcut.0.weight", name: "embedding.layer2.0.shortcut.weight", entry: "pytorch_model.updated/data/54", offset: 0,
        shape: [64, 32, 1, 1], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer2.0.shortcut.1.weight", name: "embedding.layer2.0.shortcutBN.weight", entry: "pytorch_model.updated/data/55", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.shortcut.1.bias", name: "embedding.layer2.0.shortcutBN.bias", entry: "pytorch_model.updated/data/56", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.shortcut.1.running_mean", name: "embedding.layer2.0.shortcutBN.running_mean", entry: "pytorch_model.updated/data/57",
        offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.0.shortcut.1.running_var", name: "embedding.layer2.0.shortcutBN.running_var", entry: "pytorch_model.updated/data/58",
        offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.1.conv1.weight", name: "embedding.layer2.1.conv1.weight", entry: "pytorch_model.updated/data/60", offset: 0,
        shape: [64, 64, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer2.1.bn1.weight", name: "embedding.layer2.1.bn1.weight", entry: "pytorch_model.updated/data/61", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.1.bn1.bias", name: "embedding.layer2.1.bn1.bias", entry: "pytorch_model.updated/data/62", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.1.bn1.running_mean", name: "embedding.layer2.1.bn1.running_mean", entry: "pytorch_model.updated/data/63", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.1.bn1.running_var", name: "embedding.layer2.1.bn1.running_var", entry: "pytorch_model.updated/data/64", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.1.conv2.weight", name: "embedding.layer2.1.conv2.weight", entry: "pytorch_model.updated/data/66", offset: 0,
        shape: [64, 64, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer2.1.bn2.weight", name: "embedding.layer2.1.bn2.weight", entry: "pytorch_model.updated/data/67", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.1.bn2.bias", name: "embedding.layer2.1.bn2.bias", entry: "pytorch_model.updated/data/68", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.1.bn2.running_mean", name: "embedding.layer2.1.bn2.running_mean", entry: "pytorch_model.updated/data/69", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.1.bn2.running_var", name: "embedding.layer2.1.bn2.running_var", entry: "pytorch_model.updated/data/70", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.2.conv1.weight", name: "embedding.layer2.2.conv1.weight", entry: "pytorch_model.updated/data/72", offset: 0,
        shape: [64, 64, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer2.2.bn1.weight", name: "embedding.layer2.2.bn1.weight", entry: "pytorch_model.updated/data/73", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.2.bn1.bias", name: "embedding.layer2.2.bn1.bias", entry: "pytorch_model.updated/data/74", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.2.bn1.running_mean", name: "embedding.layer2.2.bn1.running_mean", entry: "pytorch_model.updated/data/75", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.2.bn1.running_var", name: "embedding.layer2.2.bn1.running_var", entry: "pytorch_model.updated/data/76", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.2.conv2.weight", name: "embedding.layer2.2.conv2.weight", entry: "pytorch_model.updated/data/78", offset: 0,
        shape: [64, 64, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer2.2.bn2.weight", name: "embedding.layer2.2.bn2.weight", entry: "pytorch_model.updated/data/79", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.2.bn2.bias", name: "embedding.layer2.2.bn2.bias", entry: "pytorch_model.updated/data/80", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.2.bn2.running_mean", name: "embedding.layer2.2.bn2.running_mean", entry: "pytorch_model.updated/data/81", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.2.bn2.running_var", name: "embedding.layer2.2.bn2.running_var", entry: "pytorch_model.updated/data/82", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.3.conv1.weight", name: "embedding.layer2.3.conv1.weight", entry: "pytorch_model.updated/data/84", offset: 0,
        shape: [64, 64, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer2.3.bn1.weight", name: "embedding.layer2.3.bn1.weight", entry: "pytorch_model.updated/data/85", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.3.bn1.bias", name: "embedding.layer2.3.bn1.bias", entry: "pytorch_model.updated/data/86", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.3.bn1.running_mean", name: "embedding.layer2.3.bn1.running_mean", entry: "pytorch_model.updated/data/87", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.3.bn1.running_var", name: "embedding.layer2.3.bn1.running_var", entry: "pytorch_model.updated/data/88", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.3.conv2.weight", name: "embedding.layer2.3.conv2.weight", entry: "pytorch_model.updated/data/90", offset: 0,
        shape: [64, 64, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer2.3.bn2.weight", name: "embedding.layer2.3.bn2.weight", entry: "pytorch_model.updated/data/91", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.3.bn2.bias", name: "embedding.layer2.3.bn2.bias", entry: "pytorch_model.updated/data/92", offset: 0, shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.3.bn2.running_mean", name: "embedding.layer2.3.bn2.running_mean", entry: "pytorch_model.updated/data/93", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer2.3.bn2.running_var", name: "embedding.layer2.3.bn2.running_var", entry: "pytorch_model.updated/data/94", offset: 0,
        shape: [64],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.conv1.weight", name: "embedding.layer3.0.conv1.weight", entry: "pytorch_model.updated/data/96", offset: 0,
        shape: [128, 64, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.0.bn1.weight", name: "embedding.layer3.0.bn1.weight", entry: "pytorch_model.updated/data/97", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.bn1.bias", name: "embedding.layer3.0.bn1.bias", entry: "pytorch_model.updated/data/98", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.bn1.running_mean", name: "embedding.layer3.0.bn1.running_mean", entry: "pytorch_model.updated/data/99", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.bn1.running_var", name: "embedding.layer3.0.bn1.running_var", entry: "pytorch_model.updated/data/100", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.conv2.weight", name: "embedding.layer3.0.conv2.weight", entry: "pytorch_model.updated/data/102", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.0.bn2.weight", name: "embedding.layer3.0.bn2.weight", entry: "pytorch_model.updated/data/103", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.bn2.bias", name: "embedding.layer3.0.bn2.bias", entry: "pytorch_model.updated/data/104", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.bn2.running_mean", name: "embedding.layer3.0.bn2.running_mean", entry: "pytorch_model.updated/data/105", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.bn2.running_var", name: "embedding.layer3.0.bn2.running_var", entry: "pytorch_model.updated/data/106", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.shortcut.0.weight", name: "embedding.layer3.0.shortcut.weight", entry: "pytorch_model.updated/data/108", offset: 0,
        shape: [128, 64, 1, 1], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.0.shortcut.1.weight", name: "embedding.layer3.0.shortcutBN.weight", entry: "pytorch_model.updated/data/109", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.shortcut.1.bias", name: "embedding.layer3.0.shortcutBN.bias", entry: "pytorch_model.updated/data/110", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.shortcut.1.running_mean", name: "embedding.layer3.0.shortcutBN.running_mean", entry: "pytorch_model.updated/data/111",
        offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.0.shortcut.1.running_var", name: "embedding.layer3.0.shortcutBN.running_var", entry: "pytorch_model.updated/data/112",
        offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.1.conv1.weight", name: "embedding.layer3.1.conv1.weight", entry: "pytorch_model.updated/data/114", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.1.bn1.weight", name: "embedding.layer3.1.bn1.weight", entry: "pytorch_model.updated/data/115", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.1.bn1.bias", name: "embedding.layer3.1.bn1.bias", entry: "pytorch_model.updated/data/116", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.1.bn1.running_mean", name: "embedding.layer3.1.bn1.running_mean", entry: "pytorch_model.updated/data/117", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.1.bn1.running_var", name: "embedding.layer3.1.bn1.running_var", entry: "pytorch_model.updated/data/118", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.1.conv2.weight", name: "embedding.layer3.1.conv2.weight", entry: "pytorch_model.updated/data/120", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.1.bn2.weight", name: "embedding.layer3.1.bn2.weight", entry: "pytorch_model.updated/data/121", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.1.bn2.bias", name: "embedding.layer3.1.bn2.bias", entry: "pytorch_model.updated/data/122", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.1.bn2.running_mean", name: "embedding.layer3.1.bn2.running_mean", entry: "pytorch_model.updated/data/123", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.1.bn2.running_var", name: "embedding.layer3.1.bn2.running_var", entry: "pytorch_model.updated/data/124", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.2.conv1.weight", name: "embedding.layer3.2.conv1.weight", entry: "pytorch_model.updated/data/126", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.2.bn1.weight", name: "embedding.layer3.2.bn1.weight", entry: "pytorch_model.updated/data/127", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.2.bn1.bias", name: "embedding.layer3.2.bn1.bias", entry: "pytorch_model.updated/data/128", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.2.bn1.running_mean", name: "embedding.layer3.2.bn1.running_mean", entry: "pytorch_model.updated/data/129", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.2.bn1.running_var", name: "embedding.layer3.2.bn1.running_var", entry: "pytorch_model.updated/data/130", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.2.conv2.weight", name: "embedding.layer3.2.conv2.weight", entry: "pytorch_model.updated/data/132", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.2.bn2.weight", name: "embedding.layer3.2.bn2.weight", entry: "pytorch_model.updated/data/133", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.2.bn2.bias", name: "embedding.layer3.2.bn2.bias", entry: "pytorch_model.updated/data/134", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.2.bn2.running_mean", name: "embedding.layer3.2.bn2.running_mean", entry: "pytorch_model.updated/data/135", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.2.bn2.running_var", name: "embedding.layer3.2.bn2.running_var", entry: "pytorch_model.updated/data/136", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.3.conv1.weight", name: "embedding.layer3.3.conv1.weight", entry: "pytorch_model.updated/data/138", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.3.bn1.weight", name: "embedding.layer3.3.bn1.weight", entry: "pytorch_model.updated/data/139", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.3.bn1.bias", name: "embedding.layer3.3.bn1.bias", entry: "pytorch_model.updated/data/140", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.3.bn1.running_mean", name: "embedding.layer3.3.bn1.running_mean", entry: "pytorch_model.updated/data/141", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.3.bn1.running_var", name: "embedding.layer3.3.bn1.running_var", entry: "pytorch_model.updated/data/142", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.3.conv2.weight", name: "embedding.layer3.3.conv2.weight", entry: "pytorch_model.updated/data/144", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.3.bn2.weight", name: "embedding.layer3.3.bn2.weight", entry: "pytorch_model.updated/data/145", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.3.bn2.bias", name: "embedding.layer3.3.bn2.bias", entry: "pytorch_model.updated/data/146", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.3.bn2.running_mean", name: "embedding.layer3.3.bn2.running_mean", entry: "pytorch_model.updated/data/147", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.3.bn2.running_var", name: "embedding.layer3.3.bn2.running_var", entry: "pytorch_model.updated/data/148", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.4.conv1.weight", name: "embedding.layer3.4.conv1.weight", entry: "pytorch_model.updated/data/150", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.4.bn1.weight", name: "embedding.layer3.4.bn1.weight", entry: "pytorch_model.updated/data/151", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.4.bn1.bias", name: "embedding.layer3.4.bn1.bias", entry: "pytorch_model.updated/data/152", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.4.bn1.running_mean", name: "embedding.layer3.4.bn1.running_mean", entry: "pytorch_model.updated/data/153", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.4.bn1.running_var", name: "embedding.layer3.4.bn1.running_var", entry: "pytorch_model.updated/data/154", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.4.conv2.weight", name: "embedding.layer3.4.conv2.weight", entry: "pytorch_model.updated/data/156", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.4.bn2.weight", name: "embedding.layer3.4.bn2.weight", entry: "pytorch_model.updated/data/157", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.4.bn2.bias", name: "embedding.layer3.4.bn2.bias", entry: "pytorch_model.updated/data/158", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.4.bn2.running_mean", name: "embedding.layer3.4.bn2.running_mean", entry: "pytorch_model.updated/data/159", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.4.bn2.running_var", name: "embedding.layer3.4.bn2.running_var", entry: "pytorch_model.updated/data/160", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.5.conv1.weight", name: "embedding.layer3.5.conv1.weight", entry: "pytorch_model.updated/data/162", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.5.bn1.weight", name: "embedding.layer3.5.bn1.weight", entry: "pytorch_model.updated/data/163", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.5.bn1.bias", name: "embedding.layer3.5.bn1.bias", entry: "pytorch_model.updated/data/164", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.5.bn1.running_mean", name: "embedding.layer3.5.bn1.running_mean", entry: "pytorch_model.updated/data/165", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.5.bn1.running_var", name: "embedding.layer3.5.bn1.running_var", entry: "pytorch_model.updated/data/166", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.5.conv2.weight", name: "embedding.layer3.5.conv2.weight", entry: "pytorch_model.updated/data/168", offset: 0,
        shape: [128, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer3.5.bn2.weight", name: "embedding.layer3.5.bn2.weight", entry: "pytorch_model.updated/data/169", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.5.bn2.bias", name: "embedding.layer3.5.bn2.bias", entry: "pytorch_model.updated/data/170", offset: 0, shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.5.bn2.running_mean", name: "embedding.layer3.5.bn2.running_mean", entry: "pytorch_model.updated/data/171", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer3.5.bn2.running_var", name: "embedding.layer3.5.bn2.running_var", entry: "pytorch_model.updated/data/172", offset: 0,
        shape: [128],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.conv1.weight", name: "embedding.layer4.0.conv1.weight", entry: "pytorch_model.updated/data/174", offset: 0,
        shape: [256, 128, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer4.0.bn1.weight", name: "embedding.layer4.0.bn1.weight", entry: "pytorch_model.updated/data/175", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.bn1.bias", name: "embedding.layer4.0.bn1.bias", entry: "pytorch_model.updated/data/176", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.bn1.running_mean", name: "embedding.layer4.0.bn1.running_mean", entry: "pytorch_model.updated/data/177", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.bn1.running_var", name: "embedding.layer4.0.bn1.running_var", entry: "pytorch_model.updated/data/178", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.conv2.weight", name: "embedding.layer4.0.conv2.weight", entry: "pytorch_model.updated/data/180", offset: 0,
        shape: [256, 256, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer4.0.bn2.weight", name: "embedding.layer4.0.bn2.weight", entry: "pytorch_model.updated/data/181", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.bn2.bias", name: "embedding.layer4.0.bn2.bias", entry: "pytorch_model.updated/data/182", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.bn2.running_mean", name: "embedding.layer4.0.bn2.running_mean", entry: "pytorch_model.updated/data/183", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.bn2.running_var", name: "embedding.layer4.0.bn2.running_var", entry: "pytorch_model.updated/data/184", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.shortcut.0.weight", name: "embedding.layer4.0.shortcut.weight", entry: "pytorch_model.updated/data/186", offset: 0,
        shape: [256, 128, 1, 1], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer4.0.shortcut.1.weight", name: "embedding.layer4.0.shortcutBN.weight", entry: "pytorch_model.updated/data/187", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.shortcut.1.bias", name: "embedding.layer4.0.shortcutBN.bias", entry: "pytorch_model.updated/data/188", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.shortcut.1.running_mean", name: "embedding.layer4.0.shortcutBN.running_mean", entry: "pytorch_model.updated/data/189",
        offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.0.shortcut.1.running_var", name: "embedding.layer4.0.shortcutBN.running_var", entry: "pytorch_model.updated/data/190",
        offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.1.conv1.weight", name: "embedding.layer4.1.conv1.weight", entry: "pytorch_model.updated/data/192", offset: 0,
        shape: [256, 256, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer4.1.bn1.weight", name: "embedding.layer4.1.bn1.weight", entry: "pytorch_model.updated/data/193", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.1.bn1.bias", name: "embedding.layer4.1.bn1.bias", entry: "pytorch_model.updated/data/194", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.1.bn1.running_mean", name: "embedding.layer4.1.bn1.running_mean", entry: "pytorch_model.updated/data/195", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.1.bn1.running_var", name: "embedding.layer4.1.bn1.running_var", entry: "pytorch_model.updated/data/196", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.1.conv2.weight", name: "embedding.layer4.1.conv2.weight", entry: "pytorch_model.updated/data/198", offset: 0,
        shape: [256, 256, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer4.1.bn2.weight", name: "embedding.layer4.1.bn2.weight", entry: "pytorch_model.updated/data/199", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.1.bn2.bias", name: "embedding.layer4.1.bn2.bias", entry: "pytorch_model.updated/data/200", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.1.bn2.running_mean", name: "embedding.layer4.1.bn2.running_mean", entry: "pytorch_model.updated/data/201", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.1.bn2.running_var", name: "embedding.layer4.1.bn2.running_var", entry: "pytorch_model.updated/data/202", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.2.conv1.weight", name: "embedding.layer4.2.conv1.weight", entry: "pytorch_model.updated/data/204", offset: 0,
        shape: [256, 256, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer4.2.bn1.weight", name: "embedding.layer4.2.bn1.weight", entry: "pytorch_model.updated/data/205", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.2.bn1.bias", name: "embedding.layer4.2.bn1.bias", entry: "pytorch_model.updated/data/206", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.2.bn1.running_mean", name: "embedding.layer4.2.bn1.running_mean", entry: "pytorch_model.updated/data/207", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.2.bn1.running_var", name: "embedding.layer4.2.bn1.running_var", entry: "pytorch_model.updated/data/208", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.2.conv2.weight", name: "embedding.layer4.2.conv2.weight", entry: "pytorch_model.updated/data/210", offset: 0,
        shape: [256, 256, 3, 3], permutation: [0, 2, 3, 1]),
    .init(
        component: "embedding", source: "resnet.layer4.2.bn2.weight", name: "embedding.layer4.2.bn2.weight", entry: "pytorch_model.updated/data/211", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.2.bn2.bias", name: "embedding.layer4.2.bn2.bias", entry: "pytorch_model.updated/data/212", offset: 0, shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.2.bn2.running_mean", name: "embedding.layer4.2.bn2.running_mean", entry: "pytorch_model.updated/data/213", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.layer4.2.bn2.running_var", name: "embedding.layer4.2.bn2.running_var", entry: "pytorch_model.updated/data/214", offset: 0,
        shape: [256],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.seg_1.weight", name: "embedding.seg_1.weight", entry: "pytorch_model.updated/data/216", offset: 0, shape: [256, 5120],
        permutation: nil),
    .init(
        component: "embedding", source: "resnet.seg_1.bias", name: "embedding.seg_1.bias", entry: "pytorch_model.updated/data/217", offset: 0, shape: [256],
        permutation: nil)
]
