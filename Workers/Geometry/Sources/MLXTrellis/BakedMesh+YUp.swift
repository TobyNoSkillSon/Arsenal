// SPDX-License-Identifier: MIT
import MLX

extension BakedMesh {
    /// TRELLIS meshes are Z-up. Upstream o_voxel postprocess.to_glb exports Y-up for glTF (whose up axis is +Y):
    /// `y, z = z, -y` for vertices and normals. The UV flip upstream applies is undone by trimesh's own GLB export,
    /// so UVs stay as baked.
    public func yUp() -> BakedMesh {
        func swap(_ points: MLXArray) -> MLXArray {
            MLX.stacked([points[0..., 0], points[0..., 2], -points[0..., 1]], axis: 1)
        }
        return BakedMesh(
            vertices: swap(vertices), faces: faces, normals: swap(normals), uvs: uvs, texRGBA: texRGBA, atlasSize: atlasSize,
            coverage: coverage, mrRGBA: mrRGBA, filledBeforeInpaint: filledBeforeInpaint, wallFlipped: wallFlipped)
    }
}
