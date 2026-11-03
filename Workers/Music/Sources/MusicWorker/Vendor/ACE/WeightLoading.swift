//
//  WeightLoading.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  The checkpoints load as Hugging Face ships them (bf16 safetensors);
//  only layouts change here, as upstream's dit_convert.py / vae_convert.py
//  change them: PyTorch Conv1d `[out, in, k]` → MLX `[out, k, in]`,
//  ConvTranspose1d `[in, out, k]` → `[out, k, in]`, the `proj_in.1` /
//  `proj_out.1` Sequential indices dropped, weight norm (`weight_g`,
//  `weight_v`) folded into one weight, Snake parameters squeezed to `[C]`.
//  Every array must land on a parameter (`verify: .all`).
//

import Foundation
import MLX
import MLXNN

enum TorchTensorFile {

    static func silenceLatent(_ file: URL, channels: Int = 64) throws(ACEStepError) -> MLXArray {
        let data: Data
        do { data = try Data(contentsOf: file) } catch { throw .modelMissing("\(file.lastPathComponent) unreadable") }
        guard let bytes = storedEntry(in: data, suffix: "/data/0"), bytes.count % (4 * channels) == 0 else {
            throw .modelMissing("\(file.lastPathComponent) is not the expected tensor")
        }
        let frames = bytes.count / (4 * channels)
        let floats = bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        // [1, 64, T] → [T, 64]
        return MLXArray(floats, [channels, frames]).transposed(1, 0)
    }

    // The bytes of the stored (uncompressed) entry whose name ends with `suffix`.
    static func storedEntry(in data: Data, suffix: String) -> Data? {
        let b = [UInt8](data)
        func u16(_ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        guard b.count > 22 else { return nil }
        var end = b.count - 22
        while end >= 0, u32(end) != 0x0605_4B50 { end -= 1 }
        guard end >= 0 else { return nil }
        var offset = u32(end + 16)
        let count = u16(end + 10)
        for _ in 0..<count {
            guard offset + 46 <= b.count, u32(offset) == 0x0201_4B50 else { return nil }
            let method = u16(offset + 10)
            let size = u32(offset + 20)
            let nameLength = u16(offset + 28), extraLength = u16(offset + 30), commentLength = u16(offset + 32)
            let local = u32(offset + 42)
            let name = String(decoding: b[(offset + 46)..<(offset + 46 + nameLength)], as: UTF8.self)
            if name.hasSuffix(suffix), method == 0 {
                let start = local + 30 + u16(local + 26) + u16(local + 28)
                guard start + size <= b.count else { return nil }
                return Data(b[start..<(start + size)])
            }
            offset += 46 + nameLength + extraLength + commentLength
        }
        return nil
    }
}
