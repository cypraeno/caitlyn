#ifndef RENDER_H
#define RENDER_H

#include <embree4/rtcore.h>
#include "intersects.h"
#include "scene.h"
#include "vec3.h"

color trace_ray(const ray& r, std::shared_ptr<Scene> scene, int depth) {
    HitInfo record;

    color weight = color(1.0, 1.0, 1.0);
    color accumulated_color = color(0,0,0);

    ray r_in = r;

    for (int i=0; i<depth; i++) {
        ray scattered;
        color attenuation;
        struct RTCRayHit rayhit;
        setupRayHit1(rayhit, r_in);

        rtcIntersect1(scene->rtc_scene, &rayhit);
        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            std::shared_ptr<Geometry> geomhit = scene->geom_map[rayhit.hit.geomID];
            std::shared_ptr<material> mat_ptr = geomhit->materialById(rayhit.hit.geomID);
            record = geomhit->getHitInfo(r_in, r_in.at(rayhit.ray.tfar), rayhit.ray.tfar, rayhit.hit.geomID);

            color color_from_emission = mat_ptr->emitted(record.u, record.v, record.pos);
            accumulated_color += weight * color_from_emission;

            if (!mat_ptr->scatter(r_in, record, attenuation, scattered)) { // sets scattered to sampled BRDF
                return accumulated_color;
            }

            if (i == 0) {
                // Direct Light Sampling
                for (auto& light_ptr : scene->physical_lights) { // only accounts for physical lights currently
                    point3 sampled_point = light_ptr->sample(record);
                    vec3 light_dir = (sampled_point - record.pos);
                    ray sampled_ray = ray(record.pos, light_dir, 0.0);

                    // Trace a ray from here to the light
                    struct RTCRayHit light_rayhit;
                    setupRayHit1(light_rayhit, sampled_ray);

                    rtcIntersect1(scene->rtc_scene, &light_rayhit);
                    if (light_rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) { // we hit something
                        std::shared_ptr<Geometry> light_geomhit = scene->geom_map[light_rayhit.hit.geomID];
                        if (light_geomhit == light_ptr) { // if it is the light, we are not obscured from the light
                            // Store hit data of tracing the ray from here to the light
                            HitInfo light_record;
                            light_record = light_geomhit->getHitInfo(sampled_ray, sampled_ray.at(light_rayhit.ray.tfar), light_rayhit.ray.tfar, light_rayhit.hit.geomID);

                            // Get the light's material
                            std::shared_ptr<material> light_mat_ptr = light_geomhit->materialById(light_rayhit.hit.geomID);

                            ray inverse = ray(sampled_point, -light_dir, light_rayhit.ray.tfar);
                            ray light_scatter;
                            mat_ptr->scatter(inverse, record, attenuation, light_scatter);

                            // Calculate direct light contribution
                            color light_color = light_mat_ptr->emitted(light_record.u, light_record.v, light_record.pos);
                            // BRDF of light as the incoming source
                            color light_brdf = mat_ptr->generate(inverse, light_scatter, record);
                            // Calculate cos using the surface normal of the hit object and the direction from the object to the light.
                            double light_cos_theta = fmax(0.0, dot(record.normal, light_dir.unit_vector()));
                            // Evaluate the PDF considering the probability of sampling the point on the light source from the objects POV.
                            double light_pdf_value = light_ptr->pdf(light_record, sampled_ray);
                            color direct_contribution = (light_brdf * light_color * light_cos_theta) / light_pdf_value;
                            accumulated_color += weight * direct_contribution;
                        }
                    }
                }
            }

            color brdf_value = mat_ptr->generate(r_in, scattered, record);
            double pdf_value = mat_ptr->pdf(r_in, scattered, record);
            double cos_theta = fmax(0.0, dot(record.normal, scattered.direction()));

            weight = weight * ((brdf_value * cos_theta) / pdf_value);
        } else {
            // Sky background (gradient blue-white)
            vec3 unit_direction = r_in.direction().unit_vector();
            auto t = 0.5*(unit_direction.y() + 1.0);

            color sky = color(0,0,0);
            //color sky = (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
            accumulated_color += weight * sky;
            return accumulated_color;
        }

        r_in = scattered;
	}

    return accumulated_color;
}


#endif