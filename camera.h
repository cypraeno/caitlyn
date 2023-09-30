#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
class camera {
    // constructor
    public:
        __device__ camera(point3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect) {
            vec3 u, v, w;
            float theta = vfov*M_PI/180;
            float half_height = tan(theta/2);
            float half_width = aspect * half_height;

            this->origin = lookfrom;
            w = unit_vector(lookfrom - lookat);

            u = unit_vector(cross(vup, w));
            v = cross(w, u);
            this->lower_left_corner = origin - half_width*u - half_height*v - w;
            
            this->horizontal = 2*half_width*u;
            this->vertical = 2*half_height*v;
            

        }

    // methods
    public:
        /// returns ray being shot out of a camera from a certain (u,v) position
        __device__ ray get_ray(double u, double v) const {
            return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
        }
    
    // attributes
    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
};

#endif
