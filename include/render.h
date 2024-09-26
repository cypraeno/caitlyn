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
 * - MultiIntersect returns 0 length vector during direct light sampling
 * - This occurs when sampling FROM a quad, whereby the sampled point of the light yields
 * a direction PARALLEL to a u or v vector of the quad itself.
 * e.g if the quad is perpendicular to the y axis and the direction sampled is y=0, then:
 * -> MultiIntersect returns 0 length vector which normally leads to SEGFAULT.
 * -> (Unstable?) fix is added, which is to apply a small offset by the normal of the hit progressively until
 * MultiIntersect succeeds.
 * 
 * For now, direct is disabled.
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
