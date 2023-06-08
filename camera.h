#ifndef CAMERA_H
#define CAMERA_H

class camera {
    // constructor
    public:
        __device__ camera() {
            float aspect_ratio = 16.0f / 9.0f;
            float viewport_height = 2.0;
            float viewport_width = aspect_ratio * viewport_height;
            float focal_length = 1.0;

            origin = point3(0, 0, 0);
            horizontal = vec3(viewport_width, 0.0, 0.0);
            vertical = vec3(0.0, viewport_height, 0.0);
            lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);
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
