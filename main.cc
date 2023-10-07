#include "general.h"

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"


#include <iostream>
#include <chrono>

// Threading
#include <vector>
#include <thread>


hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}


// COMPILE
// g++ -std=c++11 -O2 -o renderer main.cc
// ./renderer >> latest.ppm

color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

    // if exceed bounce limit, return black (no light)
    if (depth <= 0) {
        return color(0,0,0);
    }
    
    // 0.001 instead of 0 to correct for shadow acne
    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth-1);
        return color(0,0,0);
    }

    // Sky background (gradient blue-white)
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
}

struct RenderData {
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;
    hittable_list scene;
    std::vector<color> buffer;
};


void render_scanlines(int lines, int start_line, RenderData& data, camera cam) {
    int image_width = data.image_width;
    int image_height = data.image_height;
    int samples_per_pixel = data.samples_per_pixel;
    int max_depth = data.max_depth;
    hittable_list world = data.scene;
    for (int j=start_line; j>=start_line - (lines - 1); --j) {
        for (int i=0; i<image_width; ++i) {
            color pixel_color(0, 0, 0);
            for (int s=0; s < samples_per_pixel; s++) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
                
            }
            int buffer_index = j * image_width + i;
            color buffer_pixel(pixel_color.x(),pixel_color.y(),pixel_color.z());
            data.buffer[buffer_index] = buffer_pixel;
        }
    }
}

int main() {
    std::cerr << "Press any key to begin...";
    getchar(); // Wait for user to press any key
    RenderData render_data; 

    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 50;
    const int max_depth = 50;

    render_data.image_width = image_width;
    render_data.image_height = image_height;
    render_data.samples_per_pixel = samples_per_pixel;
    render_data.max_depth = max_depth;
    render_data.buffer = std::vector<color>(image_width * image_height);
    
    // Set World
    auto world = random_scene();
    render_data.scene = world;

    // Set up Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);


    // Start Render Timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Threading approach? : Divide the scanlines into N blocks
    const int num_threads = std::thread::hardware_concurrency() - 1;
    // Image height is the number of scanlines, suppose image_height = 800
    const int lines_per_thread = image_height / num_threads;
    const int leftOver = image_height % num_threads;
    // The first <num_threads> threads are dedicated <lines_per_thread> lines, and the last thread is dedicated to <leftOver>

    std::vector<color> pixel_colors;
    std::vector<std::thread> threads;

    for (int i=0; i < num_threads; i++) {
        // In the first thead, we want the first lines_per_thread lines to be rendered
        threads.emplace_back(render_scanlines,lines_per_thread,(image_height-1) - (i * lines_per_thread),std::ref(render_data),cam);
    }
    threads.emplace_back(render_scanlines,leftOver,(image_height-1) - (num_threads * lines_per_thread),std::ref(render_data),cam);

    for (auto &thread : threads) {
            thread.join();
    }
    threads.clear();
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            int buffer_index = j * image_width + i;
            write_color(std::cout, render_data.buffer[buffer_index], samples_per_pixel);
        }
    }
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
    double time_seconds = elapsed_time / 1000.0;

    std::cerr << "\nCompleted render of scene. Render time: " << time_seconds << " seconds" << "\n";
}