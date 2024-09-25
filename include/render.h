#ifndef RENDER_H
#define RENDER_H

#include <embree4/rtcore.h>
#include "intersects.h"
#include "scene.h"
#include "vec3.h"

#include "sampling.h"

/**
 * @brief Most updated integrator for path tracing through scenes
 * 
 * 
 * @bug Known Issues:
 * - In expansive_box.csr, direct light sampling does not work -> "light_geomhit->getHitInfo(..." SEGFAULTS. 
 * There is a catch runtime error "MultiIntersect returned some id that does not exist in geom_map"
*/
color trace_ray(const ray& r, std::shared_ptr<Scene> scene, int depth);

struct RenderData {
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;
    std::vector<color> buffer;
    int completed_lines;
};

struct RayQueue {
    int index;
    int depth;
    ray r;
};

void setRenderData(RenderData& render_data, 
                    const float aspect_ratio, const int image_width,
                    const int samples_per_pixel, const int max_depth);

/** @brief recursive, shoots ray and gets its sum color through a scene. */
color colorize_ray(const ray& r, std::shared_ptr<Scene> scene, int depth);


// RENDER FUNCTIONS

void render_scanlines(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam);

void completeRayQueueTask(std::vector<RayQueue>& current, std::vector<color>& temp_buffer,
                            std::vector<color>& full_buffer, std::vector<RayQueue>& queue,
                            int mask[], int i, int current_index);

/**
 * @brief Calculates colours of the given RenderData's buffer according to the assigned lines of pixels.
 * 
 * @note for SSE 4-RayQueue packets scanline rendering
*/
void render_scanlines_sse(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam);

/**
 * @brief Calculates colours of the given RenderData's buffer according to the assigned lines of pixels.
 * 
 * @note for AVX 8-RayQueue packets scanline rendering
*/
void render_scanlines_avx(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam);

#endif
