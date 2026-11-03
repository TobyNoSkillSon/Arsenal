import MLX
// The imported graph's baseline materialization boundary; storage compression paths are deliberately absent.
enum GraniteActivationPrecision { case baseline }
enum GraniteReducedPrecisionStorage {
    static func roundTrip(_ value: MLXArray, mode: GraniteActivationPrecision) -> MLXArray { eval(value); return value }
}
