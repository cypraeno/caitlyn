#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "general.h"
#include "base.h"

/**
 * @class Camera
 * 
 * @param[in]       lookfrom Camera position
 * @param[in]       lookat point towards coordinate
 * @param[in]       vup the vector representing the normal to the camera plane. Can be specified to rotate the camera.
 * @param[in]       vfov Field of view
 * @param[in]       aspect_ratio float of width/height
 * @param[in]       focus_dist Distance from lookfrom in direction towards lookat to focus on. e.g 10.0 means the distance of highest sharpness is 10 units in direction of pointing.
 * 
 * @note Shutter open and shutter close are defaulted to 0 and 1, and have no effect on anything.
 */
class Camera : Base {
    public:
        Camera(
            point3 lookfrom,
            point3 lookat,
            vec3   vup,
            double vfov, 
            double aspect_ratio,
            double aperture,
            double focus_dist);

        ray get_ray(double s, double t) const;

    private:
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        double lens_radius;
};

#endif