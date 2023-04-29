#include "general.h"

#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include <iostream>

#include "camera.h"

// COMPILE
// g++ -std=c++11 -O2 -o renderer main.cc
// ./renderer >> latest.ppm

color ray_color(const ray& r, const hittable& world, int depth, int diffuser) {
    hit_record rec;

    if (depth <= 0) {
        return color(0,0,0);
    }
    if (world.hit(r, 0.001, infinity, rec)) {
        point3 target;
        if (diffuser == 0) {
            target = rec.p + rec.normal + random_unit_vector();
        } else if (diffuser == 1) {
            target = rec.p + random_in_hemisphere(rec.normal);
        }
        return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth-1, diffuser);
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

    // Other settings
    const int diffuse = 0; // 0 = Lambertian, 1 = Hemispheric

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
                pixel_color += ray_color(r, world, max_depth, diffuse);
            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }
    std::cerr << "\nDone.\n";
}