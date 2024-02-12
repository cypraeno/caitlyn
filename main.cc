
#include <embree4/rtcore.h>
#include "device.h"
#include "general.h"
#include "timeline.h"
#include "scene.h"
#include "camera.h"
#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "texture.h"
#include "bvh.h"


#include <iostream>
#include <chrono>


// Threading
#include <vector>
#include <thread>

auto create_still_timeline = [](vec3 pos) -> std::shared_ptr<timeline> {
    TimePosition tp1{0.0, pos, 0};
    TimePosition tp2{1.0, pos, 0};
    std::vector<TimePosition> time_positions{tp1, tp2};
    return std::make_shared<timeline>(time_positions);
};

auto create_random_timeline = [](vec3 pos) -> std::shared_ptr<timeline> {
    double y_end = random_double(pos.y(), pos.y() + 0.5);  // Adjust range as needed
    TimePosition tp1{0.0, pos, 0};
    TimePosition tp2{1.0, vec3(pos.x(), y_end, pos.z()), 0};
    std::vector<TimePosition> time_positions{tp1, tp2};
    return std::make_shared<timeline>(time_positions);
};

hittable_list random_scene() {

    hittable_list world;
    
    auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1), color(.9, .9, .9));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker), create_still_timeline(point3(0,-1000,0))));

    /*auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material, create_still_timeline(point3(0,-1000,0))));
    */

    for (int a = -11; a < 11; a++) {

        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;
                
                // diffuse material
                if (choose_mat < 0.8) {
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material, create_random_timeline(center)));
                } 

                // metal
                else if (choose_mat < 0.95) {
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material, create_random_timeline(center)));
                } 

                // glass
                else {
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material, create_random_timeline(center)));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1, create_still_timeline(point3(0,1,0))));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2, create_still_timeline(point3(-4,1,0))));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3, create_still_timeline(point3(4,1,0))));

    return world;
}

// castRay takes in an RTCScene, origin coordinates and direction coordinates of ray and casts it.
// Returns any found intersections.
bool castRay(RTCScene scene, 
             float ox, float oy, float oz,
             float dx, float dy, float dz) {
    struct RTCRayHit rayhit;
    rayhit.ray.org_x = ox;
    rayhit.ray.org_y = oy;
    rayhit.ray.org_z = oz;
    rayhit.ray.dir_x = dx;
    rayhit.ray.dir_y = dy;
    rayhit.ray.dir_z = dz;
    rayhit.ray.tnear = 0;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    // there is an alternative called rtcintersect4/8/16
    rtcIntersect1(scene, &rayhit);

    std::cout << ox << ", " << oy << ", " << oz << ": ";
    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
    std::cout << "Found intersection on geometry " << rayhit.hit.geomID << ", primitive" << rayhit.hit.primID << 
        "at tfar=" << rayhit.ray.tfar << std::endl;
    }
    else
    std::cout << "Did not find any intersection" << std::endl;
    return (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
}


hittable_list test_shutter_scene() {
    hittable_list world;

    TimePosition tp1{0.0, vec3(0, 1, 0), 0};
    TimePosition tp2{1.0, vec3(0, 2, 0), 0};
    std::vector<TimePosition> time_positions;
    time_positions.push_back(tp1);
    time_positions.push_back(tp2);
    auto timeline_ptr = std::make_shared<timeline>(time_positions);

    auto material3 = make_shared<lambertian>(color(0.1, 0.8, 0.2));
    world.add(make_shared<sphere>(point3(0, 1, 0), 0.5, material3, timeline_ptr));

    TimePosition tp3{0,vec3(0,-1000,0),0};
    TimePosition tp4{1,vec3(0,-1000,0),0};
    std::vector<TimePosition> globe_frames;
    globe_frames.push_back(tp3);
    globe_frames.push_back(tp4);
    auto timeline_ptr2 = std::make_shared<timeline>(globe_frames);
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material,timeline_ptr2));

    return world;
}

// COMPILE
// g++ -std=c++11 -O2 -o renderer main.cc
// ./renderer >> latest.ppm

struct RenderData {
    int aspect_ratio;
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;
    hittable_list scene;
    std::vector<color> buffer;
    int completed_lines;
};

 color ray_color(const ray& r, const hittable& world, int depth) {

    hit_record rec;
    // if exceed bounce limit, return black (no light)
    if (depth <= 0) return color(0,0,0);

    // 0.001 instead of 0 to correct for shadow acne
    if (world.hit(r, interval(0.001, +infinity), rec)) {
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
void setRenderData(RenderData& render_data, const float aspect_ratio, const int image_width,
    const int samples_per_pixel, const int max_depth) {
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    render_data.aspect_ratio = aspect_ratio;
    render_data.image_width = image_width;
    render_data.image_height = image_height;
    render_data.samples_per_pixel = samples_per_pixel;
    render_data.max_depth = max_depth;
    render_data.buffer = std::vector<color>(image_width * image_height);
}

void render_scanlines(int lines, int start_line, RenderData& data, camera cam) {

    int image_width         = data.image_width;
    int image_height        = data.image_height;
    int samples_per_pixel   = data.samples_per_pixel;
    int max_depth           = data.max_depth;

    hittable_list world     = data.scene;

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
            color buffer_pixel(pixel_color.x(), pixel_color.y(), pixel_color.z());
            data.buffer[buffer_index] = buffer_pixel;
        }
        data.completed_lines += 1;
        std::cerr << "[" << ((float)data.completed_lines / (float)(image_height))*100.0 << "%]" << std::endl;
    }
}

void output(RenderData& render_data, camera& cam) {
    int image_height = render_data.image_height;
    int image_width = render_data.image_width;
    int samples_per_pixel = render_data.samples_per_pixel;

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
    std::cout << "P3" << std::endl;
    std::cout << image_width << ' ' << image_height << std::endl;
    std::cout << 255 << std::endl;
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

void random_spheres() {
    // Set RenderData
    RenderData render_data; 
    const auto aspect_ratio = 3.0 / 2.0;
    setRenderData(render_data, aspect_ratio, 1200, 100, 50);

    // Set World
    auto world = random_scene();
    world = hittable_list(make_shared<bvh_node>(world));
    render_data.scene = world;
    

    // Set up Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.0001;
    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    output(render_data, cam);
}

void two_spheres() {
    // Set RenderData
    RenderData render_data; 
    const auto aspect_ratio = 16.0 / 9.0;
    setRenderData(render_data, aspect_ratio, 400, 100, 50);
    
    // Set World
    auto checker = make_shared<checker_texture>(0.8, color(.2, .3, .1), color(.9, .9, .9));
    render_data.scene.add(make_shared<sphere>(point3(0,-10, 0), 10, make_shared<lambertian>(checker), create_still_timeline(point3(0,-10,0))));
    render_data.scene.add(make_shared<sphere>(point3(0, 10, 0), 10, make_shared<lambertian>(checker), create_still_timeline(point3(0, 10,0))));
    render_data.scene = hittable_list(make_shared<bvh_node>(render_data.scene));

    // Set up Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.0001;
    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    output(render_data, cam);
}

void earth() {
    // Set RenderData
    RenderData render_data; 
    const auto aspect_ratio = 16.0 / 9.0;
    setRenderData(render_data, aspect_ratio, 400, 100, 50);

    // Set World
    auto earth_texture = make_shared<image_texture>("../images/earthmap.jpg");
    auto earth_surface = make_shared<lambertian>(earth_texture);
    auto globe = make_shared<sphere>(point3(0,0,0), 2, earth_surface, create_still_timeline(point3(0,0,0)));
    render_data.scene = hittable_list(globe);
    render_data.scene = hittable_list(make_shared<bvh_node>(render_data.scene));

    // Set up Camera
    point3 lookfrom(0,0,12);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.0001;
    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    output(render_data, cam);
}

int main() {
    switch (3) {
        case 1:  random_spheres(); break;
        case 2:  two_spheres();    break;
        case 3:  earth();          break;
    }
}

