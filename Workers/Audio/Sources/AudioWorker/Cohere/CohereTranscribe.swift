import MLX
import MLXNN
public final class CohereTranscribeModel: Module {
    public let config: CohereTranscribeConfig

    @ModuleInfo(key: "encoder") var encoder: ConformerEncoder
    @ModuleInfo(key: "decoder") var decoder: TransformerDecoderWrapper
    @ModuleInfo(key: "bridge_proj") var bridgeProj: Linear?
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: CohereTranscribeConfig) {
        self.config = config
        self._encoder.wrappedValue = ConformerEncoder(config.encoder)
        self._decoder.wrappedValue = TransformerDecoderWrapper(config: config)

        if config.encoder.dModel != config.decoder.hiddenSize {
            self._bridgeProj.wrappedValue = Linear(config.encoder.dModel, config.decoder.hiddenSize)
        }
        self._lmHead.wrappedValue = Linear(config.decoder.hiddenSize, config.vocabSize)
    }

}
