#ifndef RAY_H
#define RAY_H

#include "vec3.h"

/**
 * implementation of a ray
 * 
 * rays are of the form P(t) = A + tb, where A, b are 3D vector constants
 */
class ray {

    // constructors
    public:
        __device__ ray() {}
        __device__ ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction)
        {}

        __device__ point3 origin() const  { return orig; }
        __device__ vec3 direction() const { return dir; }

    // methods
    public:
        /**
         * calculate a given point on a ray
         * 
         * @param[in] t the independent variable of the ray
         * 
         * @returns A + tb
        */
        __device__ point3 at(float t) const {
            return orig + t*dir;
        }

    // attributes
    public:
        point3 orig;
        vec3 dir;
};

#endif
