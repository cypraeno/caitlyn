#ifndef HITINFO_H
#define HITINFO_H

#include "ray.h"

struct HitInfo {
    point3 pos;
    vec3 normal;
    vec3 microfacet_normal;
    bool front_face;
    float t;
    double u;
    double v;

    bool medium = false;

    /** @brief Given a face's outward normal and the initial ray, sets front_face to represent
    if collision hits it from the front or not. */
    void set_face_normal(const ray& r, const vec3& outward_normal);
};


#endif