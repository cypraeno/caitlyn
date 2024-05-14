#ifndef RENDER_H
#define RENDER_H

#include <embree4/rtcore.h>
#include "intersects.h"
#include "scene.h"
#include "vec3.h"

#include "sampling.h"

/**
 * @brief Most updated integrator for path tracing through scenes
 * @note Is INACCURATE WHEN BEGINNING WITHIN VOLUMES. Tracing relies are intersection with
 * volume boundary to know if it "enters" or not. If ray begins within the volume, we "enter" and never exit other than
 * doubling back.
*/
color trace_ray(const ray& r, std::shared_ptr<Scene> scene, int depth) {
    HitInfo record;

    color weight = color(1.0, 1.0, 1.0);
    color accumulated_color = color(0,0,0);

    ray r_in = r;
    BSDF_TYPE incoming_type = BSDF_TYPE::DIFFUSE;

    MediumRecord med_rec(r.origin());
    // Trace rays in all volume scenes to check which ones we reside in
    struct RTCRayHit vol_rayhit;
    HitInfo vol_record;
    for (const auto& ptr : scene->volumes) {
        setupRayHit1(vol_rayhit, r_in);
        rtcIntersect1(ptr->volume_scene, &vol_rayhit);
        int targetID;
        if (vol_rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID) {
            targetID = vol_rayhit.hit.instID[0];
        } else if (vol_rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            targetID = vol_rayhit.hit.geomID;
        } else {
            continue;
        }
        vol_record = ptr->getHitInfo(r_in, r_in.at(vol_rayhit.ray.tfar), vol_rayhit.ray.tfar, targetID);
        if (!vol_record.front_face) { // inside the volume!
            record.medium = false;
            record = vol_record;
            if (record.medium) {
                med_rec.hitVolume(ptr->medium, r_in.origin());
                incoming_type = BSDF_TYPE::SPECULAR;
            }
        }
    }

    for (int i=0; i<depth; i++) {
        // Enable of disable direct light sampling (debug only, should always be enabled)
        bool direct = true;
        bool raymarched = false; // set to true if we are colliding with a medium particle and not a surface
        std::shared_ptr<material> mat_ptr = nullptr;
        ray scattered;
        color attenuation;
        struct RTCRayHit rayhit;
        setupRayHit1(rayhit, r_in);

        rtcIntersect1(scene->rtc_scene, &rayhit);

        // Check for volume intersections
        float particleDist;
        shared_ptr<Medium> m_ptr = med_rec.particleDistance(particleDist);
        if (m_ptr) { // we are in a medium
            float istDist = rayhit.ray.tfar * r_in.direction().length();
            if (particleDist < istDist) { // volume intersection
                // Update record
                record.pos = r_in.at(particleDist / r_in.direction().length());
                raymarched = true;
            }
        }

        if (!raymarched) { // process information of next geometry hit
            int targetID;
            if (rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID) {
                targetID = rayhit.hit.instID[0];
            } else if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                targetID = rayhit.hit.geomID;
            } else {
                // Sky background (gradient blue-white)
                vec3 unit_direction = r_in.direction().unit_vector();
                auto t = 0.5*(unit_direction.y() + 1.0);

                // color sky = color(0,0,0);
                color sky = (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
                accumulated_color += weight * sky;
                return accumulated_color;
            }

            std::shared_ptr<Geometry> geomhit = scene->geom_map[targetID];
            mat_ptr = geomhit->materialById(targetID);
            record.medium = false;
            record = geomhit->getHitInfo(r_in, r_in.at(rayhit.ray.tfar), rayhit.ray.tfar, targetID);
            if (record.medium) {
                std::shared_ptr<Volume> volhit = std::dynamic_pointer_cast<Volume>(geomhit);
                if (volhit) {
                    med_rec.hitVolume(volhit->medium, r_in.at(rayhit.ray.tfar));
                    r_in = ray(r_in.at(rayhit.ray.tfar), r_in.direction(), 0.0);
                    incoming_type = BSDF_TYPE::SPECULAR;
                    continue; // ignore edges of volumes? move to next bounce
                }
            }
        } else {
            mat_ptr = m_ptr->phase;
        }
        // Get emission contribution
        color color_from_emission = mat_ptr->emitted(record.u, record.v, record.pos);

        BSDFSample sample_data = mat_ptr->sample(r_in, record, scattered);
        if (sample_data.type != BSDF_TYPE::DIFFUSE) { direct = false; }
        if (incoming_type != BSDF_TYPE::DIFFUSE || sample_data.type == BSDF_TYPE::TRANSMISSION) {
            accumulated_color += weight * color_from_emission;
        } else {
            // To prevent double contribution of emission, only directly add if and only if:
            // => we are directly hitting the light, i.e (i==0)
            if (i == 0) { accumulated_color += weight * color_from_emission; }
        }

        // Direct Light Sampling
        if (direct && color_from_emission.length() == 0.0) {
            int N = (int)scene->physical_lights.size(); // amount of lights
            for (auto& light_ptr : scene->physical_lights) { // only accounts for physical lights currently
                // Create ray from hit point to the light
                point3 sampled_point = light_ptr->sample(record);
                vec3 light_dir = (sampled_point - record.pos).unit_vector(); // direction from hit point to the light
                ray light_ray = ray(record.pos, light_dir, 0.0);

                float distWithinMedium = (sampled_point - record.pos).length(); // distance to light

                // MultiIntersect to light to capture medium boundaries
                MediumRecord light_med_rec(record.pos);
                light_med_rec.mediums = med_rec.mediums;
                light_med_rec.highest_density_volume = med_rec.highest_density_volume;
                std::vector<int> ids;
                std::vector<float> tfars;
                // errors warning: overflow in conversion from 'float' to 'int' changes value from '+Inff' to '2147483647' [-Woverflow]
                // MultiIntersect(std::numeric_limits<float>::infinity(), light_ray, scene->rtc_scene, ids, tfars);
                MultiIntersect(5, light_ray, scene->rtc_scene, ids, tfars);

                bool non_medium_encountered = false;
                std::shared_ptr<Geometry> light_geomhit;
                int light_id;
                int light_tfar;
                for (size_t j = 0; j < ids.size(); j++) {
                    int id = ids[j];
                    float tfar = tfars[j];
                    light_geomhit = scene->geom_map[id];
                    std::shared_ptr<Volume> possible_volume_hit = std::dynamic_pointer_cast<Volume>(light_geomhit);
                    if (!possible_volume_hit) { // non volume encountered
                        if (light_geomhit == light_ptr) { // hit the light
                            light_id = id;
                            light_tfar = tfar;
                            light_med_rec.hitVolume(nullptr, light_ray.at(tfar));
                            break;
                        }
                        non_medium_encountered = true;
                        break;
                    } else { // one of the intersections was a volume
                        light_med_rec.hitVolume(possible_volume_hit->medium, light_ray.at(tfar));
                    }
                }
               
                if (!non_medium_encountered) { // if it is the light, we are not obscured from the light
                    // Store hit data of tracing the ray from here to the light
                    HitInfo light_record;
                    light_record = light_geomhit->getHitInfo(light_ray, light_ray.at(light_tfar), light_tfar, light_id);
                    
                    // Get the light's material
                    std::shared_ptr<material> light_mat_ptr = light_geomhit->materialById(light_id);

                    // Sample BSDF of hit point with incoming light
                    ray light_scattered;
                    
                    BSDFSample light_sample_data;
                    color att;
                    // We do NOT call the above line because it would sample a possibly different microfacet normal
                    // than what is already sampled previous to the Direct Light Sampling (for complex BSDFs that use microfacets)
                    // Both generate and pdf assume that, if a microfacet normal is needed, it is already defined. Thus, we use the previous
                    // and pass in the same HitInfo.
                    light_sample_data.bsdf_value = mat_ptr->generate(r_in, light_ray, record);
                    light_sample_data.pdf_value = mat_ptr->pdf(r_in, light_ray, record);
                    
                    // Find pdf for the light hit point
                    double light_pdf_value = light_ptr->pdf(light_record, light_ray);

                    // Find contribution of light using MIS power heuristic of light_pdf and sample pdf
                    double light_cos_theta = fabs(dot(record.normal, light_dir));
                    color light_contribution = weight * light_sample_data.bsdf_value * light_cos_theta
                            * MIS::power_heuristic<MIS::EVAL_WEIGHT>(light_pdf_value, light_sample_data.pdf_value) / light_pdf_value;
                    float transmittance_coeff = light_med_rec.transmittance;
                    light_contribution *= transmittance_coeff;
                    // Get emission of light
                    color light_Le = light_mat_ptr->emitted(light_record.u, light_record.v, light_record.pos);
                    accumulated_color += light_contribution * light_Le / N;
                }
            }
        }

        // Indirect ray contribution
        if (!sample_data.scatter) {
            return accumulated_color;
        }
        double cos_theta = fabs(dot(record.normal, (sample_data.scatter_direction)));
        if (!raymarched) {
            weight = weight * (sample_data.bsdf_value * cos_theta / sample_data.pdf_value);
        } else {
            weight = weight * (sample_data.bsdf_value / sample_data.pdf_value);
        }
        r_in = scattered;
        incoming_type = sample_data.type;
	}
    return accumulated_color;
}

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
