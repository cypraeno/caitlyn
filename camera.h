#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "general.h"

class camera {
    public:
        camera(
            point3 lookfrom,
            point3 lookat,
            vec3   vup,
            double vfov, 
            double aspect_ratio,
            double aperture,
            double focus_dist,
            double _time0 = 0,
            double _time1 = 1) {
            
            auto theta = degrees_to_radians(vfov);
            auto h = tan(theta/2);
            auto viewport_height = 2.0 * h;
            auto viewport_width = aspect_ratio * viewport_height;
            
            w = (lookfrom - lookat).unit_vector();
            u = (cross(vup, w)).unit_vector();
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

            lens_radius = aperture / 2;
            time0 = _time0;
            time1 = _time1;
        }

        ray get_ray(double s, double t) const {
            vec3 rd = lens_radius * random_in_unit_disk();
            vec3 offset = u * rd.x() + v * rd.y();


            return ray(origin + offset, 
                lower_left_corner + s*horizontal + t*vertical - origin - offset,
                random_double(time0,time1));
        }

        color ray_color(const ray& r, const hittable& world, int depth) {

            hit_record rec;
            // if exceed bounce limit, return black (no light)
            if (depth <= 0) {
                return color(0,0,0);
            }

            // 0.001 instead of 0 to correct for shadow acne
            if (world.hit(r, interval(0.001, +infinity) rec)) {
                ray scattered;
                color attenuation;

                if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) 
                    return attenuation * ray_color(scattered, world, depth-1);

                return color(0,0,0);
            }

            // Sky background (gradient blue-white)
            vec3 unit_direction = r.direction().unit_vector();
            auto t = 0.5*(unit_direction.y() + 1.0);

            return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
        }


    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        double lens_radius;
        double time0, time1; // shutter open -> shutter close
};

#endif