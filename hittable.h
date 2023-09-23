#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

class material;


/*
 * details of a hit in a scene
 * 
 * p            point hit
 * normal       direction of the normal from p
 * mat          material of the object
 * t            length of the ray when it hits p
 */
struct hit_record {
    point3 p;
    vec3 normal;
    material mat;
    float t;
};


/**
 * abstraction of a hittable object in a scene
 */
class hittable {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};


#endif
