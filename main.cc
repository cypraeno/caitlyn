#include <embree4/rtcore.h>
#include "device.h"
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


RTCScene initializeScene(RTCDevice device) {
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vertices = (float*) rtcSetNewGeometryBuffer(geom,
                                                     RTC_BUFFER_TYPE_VERTEX,
                                                     0,
                                                     RTC_FORMAT_FLOAT3,
                                                     3*sizeof(float),
                                                     3);

    unsigned* indices = (unsigned*) rtcSetNewGeometryBuffer(geom,
                                                          RTC_BUFFER_TYPE_INDEX,
                                                          0,
                                                          RTC_FORMAT_UINT3,
                                                          3*sizeof(unsigned),
                                                          1);
    // rtcSetNewGeometryBuffer creates data buffer. set of three vertices make a point, three points make a triangle. 
    // stored in 1D array, indices allocated to make it easy to access the values in the 1D array instead of 2D
    // assuming it's orderred like [x, ... ,x,y, ... ,y,z, ... ,z]
    // formula to get a certay x,y,z point is indices[point_ix] + indices.length()*dimension
    // for i >= 0, where point_ix is the intended 'point' from indices, and dimension ls 0,1,2, for x,y,z respectively

    if (vertices && indices) {
        vertices[0] = 0.f; vertices[1] = 0.f; vertices[2] = 0.f;
        vertices[3] = 1.f; vertices[4] = 0.f; vertices[5] = 0.f;
        vertices[6] = 0.f; vertices[7] = 1.f; vertices[8] = 0.f;

        indices[0] = 0; indices[1] = 1; indices[2] = 2;
    }

    RTCGeometry sphere = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
    float* spherev = (float*)rtcSetNewGeometryBuffer(sphere, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, 4*sizeof(float), 1);
    if (spherev) {
        spherev[0] = 0;
        spherev[1] = 0;
        spherev[2] = 0;
        spherev[3] = 1;
    }

    rtcCommitGeometry(sphere);
    unsigned int sphereID = rtcAttachGeometry(scene, sphere);
    rtcReleaseGeometry(sphere);

    rtcCommitScene(scene);

    return scene;
}

void castRay(RTCScene scene, 
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
}



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
                
                // diffuse material
                if (choose_mat < 0.8) {
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } 

                // metal
                else if (choose_mat < 0.95) {
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } 

                // glass
                else {
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

        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) return attenuation * ray_color(scattered, world, depth-1);

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
    }
}

int main() {
    RTCDevice device = initializeDevice();
    RTCScene scene = initializeScene(device);

    /* This will hit the triangle at t=1. */
    castRay(scene, 0.33f, 0.33f, -1, 0, 0, 1);

    /* This will not hit anything. */
    castRay(scene, 1.00f, 1.00f, -1, 0, 0, 1);

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
    
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