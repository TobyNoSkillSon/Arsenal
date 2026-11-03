#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef int (*GeometryCancel)(void *);
typedef struct {
    float *positions;
    float *uvs;
    uint32_t *indices;
    uint32_t *vertex_map;
    uint32_t vertex_count, index_count, chart_count, width, height;
} GeometryAtlas;
int geometry_atlas_generate(const float *positions, uint32_t vertex_count, const uint32_t *indices,
    uint32_t index_count, uint32_t resolution, GeometryCancel cancelled, void *context, GeometryAtlas *result);
void geometry_atlas_free(GeometryAtlas *result);
#ifdef __cplusplus
}
#endif
