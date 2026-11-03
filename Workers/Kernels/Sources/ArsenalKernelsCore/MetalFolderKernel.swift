// SPDX-License-Identifier: MIT
import Foundation
import Metal

/// An `api = "metal"` kernel for engines that drive Metal themselves (own engine, C/C++/Rust through a shim):
/// the folder's complete MSL source compiled at runtime into a pipeline for `entry`. A compile failure is a
/// result, never a crash; it is persisted per (digest, OS build, GPU) so the next load goes straight to stock.
public enum MetalFolderKernel {
    public static func pipeline(
        _ kernel: LoadedKernel, maxThreadsPerThreadgroup: Int = 1,
        compiler: KernelCompiler = .shared, verdicts: KernelVerdicts = .shared
    ) -> Result<MTLComputePipelineState, KernelError> {
        guard kernel.spec.api == .metal else { return .failure(.invalid("Kernel \(kernel.spec.id) is not an api = \"metal\" kernel.")) }
        let key = KernelVerdicts.key("metal-compile", kernel.digest, kernel.spec.entry, String(maxThreadsPerThreadgroup))
        if let saved = verdicts.read(key), !saved.pass { return .failure(.invalid(saved.detail)) }
        if let failure = compiler.check(source: kernel.body, function: kernel.spec.entry, threadsPerThreadgroup: maxThreadsPerThreadgroup) {
            verdicts.write(key, .init(pass: false, detail: failure))
            return .failure(.invalid(failure))
        }
        do {
            let library = try compiler.library(source: kernel.body)
            guard let function = library.makeFunction(name: kernel.spec.entry), let device = compiler.device else {
                return .failure(.invalid("The library has no function \(kernel.spec.entry)."))
            }
            let pipeline = try device.makeComputePipelineState(function: function)
            verdicts.write(key, .init(pass: true, detail: "compiled"))
            return .success(pipeline)
        } catch {
            return .failure(.invalid("Metal pipeline failed: \(error)"))
        }
    }
}
