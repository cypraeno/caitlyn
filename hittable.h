#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"


/*
 * details of a hit in a scene
 * 
 * p            point hit
 * normal       direction of the normal from p
 * t            length of the ray when it hits p
 * front_face   if object is front-facing
 */
struct hit_record {
    point3 p;
    vec3 normal;
    float t;
    bool front_face;

    /// sets the values for normal and front_face 
    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


/**
 * abstraction of a hittable object in a scene
 */
class hittable {
    public:
        __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};


#endif
