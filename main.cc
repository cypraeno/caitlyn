#include "general.h"
#include "camera.h"

#include "color.h"
#include "hittable_list.h"
#include "sphere.h"

#include <iostream>


// COMPILE
// g++ -std=c++11 -O2 -o my_renderer main.cc

color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

    // if exceed bounce limit, return black (no light)
    if (depth <= 0) {
        return color(0, 0, 0);
    }

    // 0.001 instead of 0 to correct for shadow acne
    if (world.hit(r, 0.001, infinity, rec)) {
        // 3 different diffusion options (uncomment desired one)

        // point3 target = rec.p + rec.normal + random_in_unit_sphere();            // Rejection Method
        point3 target = rec.p + rec.normal + random_unit_vector();               // Lambertian Diffuse
        // point3 target = rec.p + random_in_hemisphere(rec.normal);                // Hemispherical Scattering
        
        return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);    // assign colour based off of ray from point to target
    }

    // Sky background (gradient blue-white)
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
}

int main() {
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int max_depth = 50;

    hittable_list world;
    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));   

    camera cam;

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            color pixel_color(0, 0, 0);
            for (int s=0; s < samples_per_pixel; s++) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }
    std::cerr << "\nDone.\n";
}