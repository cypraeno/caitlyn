#include "material.h"

#include "medium.h"

BSDFSample LayeredBSDF::sample(const ray& r_in, HitInfo& rec, ray& scattered) const {
    // Return in case calculating a full simulation becomes impossible or irrelevant
    BSDFSample absorbed; absorbed.scatter = false;
    
    ray r = r_in;
    HitInfo rec_manip = rec;
    vec3 outward_normal = rec.front_face ? rec.normal : -rec.normal;

    bool on_top = rec_manip.front_face;
    rec_manip.front_face ? rec_manip.normal : -rec_manip.normal;

    std::shared_ptr<material> current = on_top ? top : bottom;
    BSDFSample bs = current->sample(r, rec_manip, scattered);

    color f = bs.bsdf_value;
    float pdf = bs.pdf_value;

    float distance_in_from_top = rec.front_face ? 0.0 : 1.0; // amount of distance ray has traveled into between the layers
    bool prev_medium = false; // previous hit was a medium

    if (!bs.scatter) { return absorbed; }
    if (bs.type != BSDF_TYPE::TRANSMISSION && bs.type != BSDF_TYPE::TRANSPARENT) {
        bs.type = (dot(bs.scatter_direction, rec.normal) < 0) ? BSDF_TYPE::TRANSMISSION : BSDF_TYPE::SPECULAR;
        return bs;
    }
    for (int depth = 0; depth < termination; depth++) {
        
        // Follow random walk through layers to sample layered BSDF
        on_top = !on_top;
        current = on_top ? top : bottom;

        // Possibly terminate layered BSDF sampling with Russian Roulette
        float rrBeta = fmax(fmax(f.x(), f.y()), f.z()) / bs.pdf_value;
        if (depth > 3 && rrBeta < 0.25) { // rrBeta < 0.50 reduces by more, but probably reduces accuracy
            float q = fmax(0, 1-rrBeta);
            if (random_double() < q) {
                if (on_top == rec.front_face) {
                    bs.type = (dot(bs.scatter_direction, rec.normal) < 0) ? BSDF_TYPE::TRANSMISSION : BSDF_TYPE::SPECULAR;
                    return bs;
                }
            }
            pdf *= 1 - q;
        }

        if (!prev_medium) { f = f * fabs(dot(rec_manip.normal, (bs.scatter_direction))); }
        prev_medium = false;
        r = ray(rec_manip.pos - bs.scatter_direction, bs.scatter_direction, r.time());

        if (medium) {
            float distance = medium->particleDistance();
            float distance_to_barrier;
            if (on_top) { distance_to_barrier = distance_in_from_top; }
            else { distance_to_barrier = thickness - distance_in_from_top; }
            if (distance < distance_to_barrier) { // collide with particle
                distance_in_from_top += (on_top * -distance) + (!on_top * distance);
                bs = medium->phase->sample(r, rec_manip, scattered);
                prev_medium = true;

                f = f * bs.bsdf_value;
                pdf = pdf * bs.pdf_value;
                bs.bsdf_value = f;
                bs.pdf_value = pdf;

                if (dot(bs.scatter_direction, outward_normal) > 0) {
                    on_top = false; // because it will flip at the start of the next loop
                    rec_manip.front_face = false;
                    rec_manip.normal = -outward_normal;
                } else {
                    on_top = true; // because it will flip at the start of the next loop
                    rec_manip.front_face = true;
                    rec_manip.normal = outward_normal;
                }

                continue;
            } else {
                if (on_top) { distance_in_from_top = 0.0;} else { distance_in_from_top = 1.0; }
            }
        }

        bs = current->sample(r, rec_manip, scattered);
        f = f * bs.bsdf_value;
        pdf = pdf * bs.pdf_value;
        bs.bsdf_value = f;
        bs.pdf_value = pdf;

        if (on_top && bs.type == BSDF_TYPE::TRANSMISSION) {
            bs.type = (dot(bs.scatter_direction, rec.normal) < 0) ? BSDF_TYPE::TRANSMISSION : BSDF_TYPE::SPECULAR;
            return bs;
        }

        // Flip since coming from the bottom!
        rec_manip.front_face = false;
        rec_manip.normal = -rec_manip.normal;
    }
    bs.type = (dot(bs.scatter_direction, rec.normal) < 0) ? BSDF_TYPE::TRANSMISSION : BSDF_TYPE::SPECULAR;
    return bs;
}