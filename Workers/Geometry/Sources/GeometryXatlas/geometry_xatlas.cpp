// SPDX-License-Identifier: MIT
#include "include/geometry_xatlas.h"
#include "xatlas.h"
#include <cstdlib>
#include <cstring>
#include <memory>
struct Progress { GeometryCancel cancelled; void *context; };
static bool progress(xatlas::ProgressCategory, int, void *opaque) {
    auto *state = static_cast<Progress *>(opaque);
    return !state->cancelled || !state->cancelled(state->context);
}
extern "C" int geometry_atlas_generate(const float *positions, uint32_t vertex_count, const uint32_t *indices,
    uint32_t index_count, uint32_t resolution, GeometryCancel cancelled, void *context, GeometryAtlas *result) {
    xatlas::SetPrint(nullptr, false);
    std::memset(result, 0, sizeof(*result));
    if (!positions || !indices || !vertex_count || index_count < 3 || index_count % 3) return 1;
    std::unique_ptr<xatlas::Atlas, decltype(&xatlas::Destroy)> atlas(xatlas::Create(), xatlas::Destroy);
    if (!atlas) return 1;
    Progress state{cancelled, context};
    xatlas::SetProgressCallback(atlas.get(), progress, &state);
    xatlas::MeshDecl input;
    input.vertexPositionData = positions; input.vertexCount = vertex_count; input.vertexPositionStride = 3 * sizeof(float);
    input.indexData = indices; input.indexCount = index_count; input.indexFormat = xatlas::IndexFormat::UInt32;
    if (xatlas::AddMesh(atlas.get(), input) != xatlas::AddMeshError::Success) return 1;
    if (cancelled && cancelled(context)) return 2;
    xatlas::PackOptions options; options.resolution = resolution; options.padding = 2;
    xatlas::Generate(atlas.get(), xatlas::ChartOptions(), options);
    if (cancelled && cancelled(context)) return 2;
    if (atlas->meshCount != 1 || !atlas->width || !atlas->height) return 1;
    const auto &mesh = atlas->meshes[0];
    if (!mesh.vertexCount || !mesh.indexCount) return 1;
    result->vertex_count = mesh.vertexCount; result->index_count = mesh.indexCount;
    result->chart_count = mesh.chartCount; result->width = atlas->width; result->height = atlas->height;
    result->positions = static_cast<float *>(std::malloc(mesh.vertexCount * 3 * sizeof(float)));
    result->uvs = static_cast<float *>(std::malloc(mesh.vertexCount * 2 * sizeof(float)));
    result->vertex_map = static_cast<uint32_t *>(std::malloc(mesh.vertexCount * sizeof(uint32_t)));
    result->indices = static_cast<uint32_t *>(std::malloc(mesh.indexCount * sizeof(uint32_t)));
    if (!result->positions || !result->uvs || !result->vertex_map || !result->indices) { geometry_atlas_free(result); return 1; }
    for (uint32_t i = 0; i < mesh.vertexCount; ++i) {
        const auto &vertex = mesh.vertexArray[i];
        if (vertex.xref >= vertex_count) { geometry_atlas_free(result); return 1; }
        result->vertex_map[i] = vertex.xref;
        std::memcpy(result->positions + i * 3, positions + vertex.xref * 3, 3 * sizeof(float));
        result->uvs[i * 2] = vertex.uv[0] / atlas->width;
        result->uvs[i * 2 + 1] = vertex.uv[1] / atlas->height;
    }
    std::memcpy(result->indices, mesh.indexArray, mesh.indexCount * sizeof(uint32_t));
    return 0;
}
extern "C" void geometry_atlas_free(GeometryAtlas *result) {
    std::free(result->positions); std::free(result->uvs); std::free(result->vertex_map); std::free(result->indices);
    std::memset(result, 0, sizeof(*result));
}
