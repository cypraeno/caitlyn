#include "camera.h"

Camera::Camera(
    point3 lookfrom,
    point3 lookat,
    vec3   vup,
    double vfov, 
    double aspect_ratio,
    double aperture,
    double focus_dist) : Base(lookfrom) {
    
    // Convert vertical field of view from degrees to radians
    auto theta = degrees_to_radians(vfov);
    // Calculate the height of the viewport
    auto h = tan(theta/2);
    auto viewport_height = 2.0 * h;
    // Calculate the width of the viewport
    auto viewport_width = aspect_ratio * viewport_height;
    
    // Calculate the camera's orthonormal basis vectors for orientation
    w = (position - lookat).unit_vector();  // Viewing direction vector (reverse)
    u = (cross(vup, w)).unit_vector();      // Right-side vector of the camera
    v = cross(w, u);                        // Actual "up" vector for the camera

    // Calculate the vectors representing the viewport dimensions
    horizontal = focus_dist * viewport_width * u;  // Horizontal vector scaled by focus distance
    vertical = focus_dist * viewport_height * v;   // Vertical vector scaled by focus distance
    // Calculate the lower-left corner of the viewport
    lower_left_corner = position - horizontal/2 - vertical/2 - focus_dist * w;

    // Calculate the radius of the lens for depth of field effects
    lens_radius = aperture / 2;
}

ray Camera::get_ray(double s, double t) const {
    vec3 rd = lens_radius * random_in_unit_disk();
    vec3 offset = u * rd.x() + v * rd.y();


    return ray(position + offset, 
        lower_left_corner + s*horizontal + t*vertical - position - offset, 0.0);
}