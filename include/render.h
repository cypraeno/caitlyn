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
            color brdf_value = mat_ptr->generate(r_in, scattered, record);
            double pdf_value = mat_ptr->pdf(r_in, scattered, record);
            double cos_theta = fmax(0.0, dot(record.normal, scattered.direction()));

            weight = weight * ((brdf_value * cos_theta) / pdf_value);

            // direct light checks occur here.
        } else {
            // Sky background (gradient blue-white)
            vec3 unit_direction = r_in.direction().unit_vector();
            auto t = 0.5*(unit_direction.y() + 1.0);

            color sky = (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
            accumulated_color += weight * sky;
            return accumulated_color;
        }

        r_in = scattered;
	}

    return accumulated_color;
}


#endif