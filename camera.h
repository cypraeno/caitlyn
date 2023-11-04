#ifndef CAMERA_H
#define CAMERA_H

#include <curand_kernel.h>
#include "ray.h"
#include "vec3.h"

/**
 * produce a random point in a unit disk
 * 
 * @param[in] local_rand_state the CUDA random state
 * 
 * @return a random point
 */
__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0*vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

/**
 * implementation of a camera class
 */
class camera {
    // constructor
    public:
        __device__ camera(point3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperature, float focus_dist) {
            this->lens_radius = aperature / 2.0f;
            
            // determines the height and width of the camera
            float theta = vfov*((float)M_PI)/180.0f;
            float half_height = tan(theta/2);
            float half_width = aspect * half_height;
            this->horizontal = 2.0f*half_width*focus_dist*u;
            this->vertical = 2.0f*half_height*focus_dist*v;

            // determine the origin and direction of the camer
            this->origin = lookfrom;
            this->w = unit_vector(lookfrom - lookat);
            
            // determine lower left corner of the camera
            this->u = unit_vector(cross(vup, w));
            this->v = cross(w, u);
            this->lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
        }

    // methods
    public:
        /**
         * returns ray being shot out of a camera from a certain (u,v) position
         * 
         * @param[in] s
         * @param[in] t
         * @param[in] local_rand_state
         * 
         * @return the ray being shot out
         */
        __device__ ray get_ray(float s, float t, curandState *local_rand_state) const {
            vec3 rd = lens_radius*random_in_unit(local_rand_state);
            vec3 offset = u*rd.x() + v*rd.();
            return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
        }
    
    // attributes
    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        float lens_radius;
};

#endif
