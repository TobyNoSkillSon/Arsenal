// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN

final class RetrievalGraph: Module {
    @ModuleInfo(key: "model") var backbone: EmbedQwenModelInner
    @ModuleInfo(key: "lm_head") var head: Linear?
    @ModuleInfo(key: "projector") var projector: JinaProjector?
    init(_ config: EmbedQwenConfiguration, ranking: Bool, jina: Bool) {
        _backbone.wrappedValue = EmbedQwenModelInner(config)
        if ranking && !jina && !config.tieWordEmbeddings {
            _head.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        if jina { _projector.wrappedValue = JinaProjector(config.hiddenSize) }
    }
    func logits(_ hidden: MLXArray) -> MLXArray {
        if let head { return head(hidden) }
        return backbone.embedTokens.asLinear(hidden)
    }
}

final class JinaProjector: Module {
    @ModuleInfo(key: "linear1") var first: Linear
    @ModuleInfo(key: "linear2") var second: Linear
    init(_ size: Int) {
        _first.wrappedValue = Linear(size, 512, bias: false)
        _second.wrappedValue = Linear(512, 512, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { second(relu(first(x))) }
}
