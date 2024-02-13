#include <embree4/rtcore.h>
#include "device.h"
#include "general.h"
#include "scene.h"
#include "camera.h"
#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "material.h"
#include "sphere_primitive.h"
#include "intersects.h"


#include <iostream>
#include <chrono>


// Threading
#include <vector>
#include <thread>

/** @brief Create a standardized scene benchmark for testing optimizations between different versions 
 * 
 * @param shared_ptr<Scene> scene_ptr Pointer to the scene that will be modified.
 * @param RTCDevice device object for instantiation. must not be released yet.
 * @note Benchmark v0.1.0
 * @note Standard benchmark scene creates a large ground sphere with 1000 radius, at 0,-1000,0
 * @note Then instantiate 22*22 sphere. In each iteration, choose randomized position and material.
*/
void setup_benchmark_scene(std::shared_ptr<Scene> scene_ptr, RTCDevice device) {
    std::cerr << "Setup Benchmark Scene v0.1.0" << std::endl;
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    auto ground_sphere = make_shared<SpherePrimitive>(point3(0,-1000,0), ground_material, 1000, device);
    unsigned int groundID = scene_ptr->add_primitive(ground_sphere);
    std::cerr << "ADD PRIM :: (0,-1000,0), RADIUS 1000, LAMBERTIAN" << std::endl;

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
                } 

                // metal
                else if (choose_mat < 0.95) {
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                } 

                // glass
                else { sphere_material = make_shared<dielectric>(1.5); }
                
                auto sphere = make_shared<SpherePrimitive>(center, sphere_material, 0.2, device);
                scene_ptr->add_primitive(sphere);
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    auto sphere1 = make_shared<SpherePrimitive>(point3(0, 1, 0), material1, 1, device);
    scene_ptr->add_primitive(sphere1);

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    auto sphere2 = make_shared<SpherePrimitive>(point3(-4, 1, 0), material2, 1, device);
    scene_ptr->add_primitive(sphere2);

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    auto sphere3 = make_shared<SpherePrimitive>(point3(4, 1, 0), material3, 1, device);
    scene_ptr->add_primitive(sphere3);

    // Finalizing the Scene
    scene_ptr->commitScene();
    std::cerr << "COMMIT SCENE :: complete" << std::endl;
}

// COMPILE
// g++ -std=c++11 -O2 -o renderer main.cc
// ./renderer >> latest.ppm

/** @brief recursive, shoots ray and gets its sum color through a scene. */
color colorize_ray(const ray& r, std::shared_ptr<Scene> scene, int depth) {
    HitInfo record;

    // end of recursion
    if (depth <= 0) {
        return color(0,0,0);
    }

    // fire ray into scene and get ID.
    struct RTCRayHit rayhit;
    setupRayHit1(rayhit, r);

    rtcIntersect1(scene->rtc_scene, &rayhit);

    // if hit is found
    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        ray scattered;
        color attenuation;

        // get the material of the thing we just hit
        std::shared_ptr<Geometry> geomhit = scene->geom_map[rayhit.hit.geomID];
        std::shared_ptr<material> mat_ptr = geomhit->materialById(rayhit.hit.geomID);
        record = geomhit->getHitInfo(r, r.at(rayhit.ray.tfar), rayhit.ray.tfar, rayhit.hit.geomID);
        if (mat_ptr->scatter(r, record, attenuation, scattered)) return attenuation * colorize_ray(scattered, scene, depth-1);

        return color(0,0,0);
    }

    // Sky background (gradient blue-white)
    vec3 unit_direction = r.direction().unit_vector();
    auto t = 0.5*(unit_direction.y() + 1.0);

    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
}

struct RenderData {
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;
    std::vector<color> buffer;
    int completed_lines;
};


void render_scanlines(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam) {

    int image_width         = data.image_width;
    int image_height        = data.image_height;
    int samples_per_pixel   = data.samples_per_pixel;
    int max_depth           = data.max_depth;

    for (int j=start_line; j>=start_line - (lines - 1); --j) {

        for (int i=0; i<image_width; ++i) {

            color pixel_color(0, 0, 0);

            for (int s=0; s < samples_per_pixel; s++) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                pixel_color += colorize_ray(r, scene_ptr, max_depth);
            }

            int buffer_index = j * image_width + i;
            color buffer_pixel(pixel_color.x(), pixel_color.y(), pixel_color.z());
            data.buffer[buffer_index] = buffer_pixel;
        }
        data.completed_lines += 1;
        float percentage_completed = ((float)data.completed_lines / (float)data.image_height)*100.00;
        std::cerr << "[" <<int(percentage_completed) << "%] completed" << std::endl;
    }
}

struct RayQueue {
    int index;
    int depth;
    ray r;
};
void render_scanlines_sse(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam) {
    int image_width         = data.image_width;
    int image_height        = data.image_height;
    int samples_per_pixel   = data.samples_per_pixel;
    int max_depth           = data.max_depth;
    
    for (int j=start_line; j>=start_line - (lines - 1); --j) {
        std::vector<color> full_buffer(image_width);
        for (int s=0; s < samples_per_pixel; s++) {
            std::vector<RayQueue> queue;
            std::vector<color> temp_buffer(image_width);
            std::vector<RayQueue> current(4); // size = 4 only
            for (int i=image_width-1; i>=0; --i) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                
                RayQueue q = { i, 0, r };
                queue.push_back(q);
            }

            RTCRayHit4 rayhit;

            for (int i=0; i<4; i++) {
                RayQueue back = queue.back();
                queue.pop_back();
                current.push_back(back);
            }

            int current_index_last_completed = -1;
            int mask[4] = {-1, -1, -1, -1};
            while (!queue.empty()) {
                std::vector<ray> rays;
                for (int i=0; i<(int)current.size(); i++) {
                    rays.push_back(current[i].r);
                }
                setupRayHit4(rayhit, rays);
                rtcIntersect4(mask, scene_ptr->rtc_scene, &rayhit);
                
                HitInfo record;

                for (int i=0; i<4; i++) {
                    ray current_ray = current[i].r;
                    int current_index = current[i].index;
                    // process each ray by editing the temp_buffer and updating current queue
                    if (rayhit.hit.geomID[i] != RTC_INVALID_GEOMETRY_ID) { // hit
                        ray scattered;
                        color attenuation;

                        std::shared_ptr<Geometry> geomhit = scene_ptr->geom_map[rayhit.hit.geomID[i]];
                        std::shared_ptr<material> mat_ptr = geomhit->materialById(rayhit.hit.geomID[i]);
                        record = geomhit->getHitInfo(current_ray, current_ray.at(rayhit.ray.tfar[i]), rayhit.ray.tfar[i], rayhit.hit.geomID[i]);
                        if (mat_ptr->scatter(current_ray, record, attenuation, scattered)) {
                            temp_buffer[current_index] = temp_buffer[current_index] * attenuation;
                            
                            if (current[i].depth + 1 == max_depth) { // reached max depth, replace with next in queue
                                // check if theres even any more to do, if not then break out.
                                if ((int)queue.size() >= 1) { // at least one remaining
                                    // replace finished RayQueue with next
                                    RayQueue back = queue.back();
                                    queue.pop_back();
                                    current[i] = back;
                                } else { // no more remaining
                                    mask[i] = 0; // disable this part of the packet from running
                                }
                            } else { // not finished depth wise
                                current[i].depth += 1;
                                current[i].r = scattered;
                            }
                        }

                    } else { // no hit
                        // Sky background (gradient blue-white)
                        vec3 unit_direction = current_ray.direction().unit_vector();
                        auto t = 0.5*(unit_direction.y() + 1.0);

                        color multiplier = (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
                        temp_buffer[current_index] = temp_buffer[current_index] * multiplier;
                        
                        // check if theres even any more to do, if not then break out.
                        if ((int)queue.size() >= 1) { // at least one remaining
                            // replace finished RayQueue with next
                            RayQueue back = queue.back();
                            queue.pop_back();
                            current[i] = back;
                        } else { // no more remaining
                            mask[i] = 0;
                        }
                    }
                }
            }
            /*
            if (current_index_last_completed != -1) {
                // if this runs, it means we reached a max depth or didn't hit anything on the ith
                // block of the current queue. Thus, we call colorize_ray on the remaining.
                // It means that in block size N, there are N - 1 pixels that are still going;
                // all the onesb before ith, and all the ones after.
                for (int i=0; i<4; i++) {
                    if (i != current_index_last_completed) {
                        ray current_ray = current[i].r;
                        int current_index = current[i].index;
                        temp_buffer[current_index] = colorize_ray(current_ray, scene_ptr, max_depth);
                    }
                }
            }
            */
            for (int i=0; i<image_width; ++i) {
                full_buffer[i] += temp_buffer[i];
            }
        }
        for (int i=0; i<image_width; ++i) {
            int buffer_index = j * image_width + i;
            color buffer_pixel(full_buffer[i].x(), full_buffer[i].y(), full_buffer[i].z());
            data.buffer[buffer_index] = buffer_pixel;
        }
        data.completed_lines += 1;
        float percentage_completed = ((float)data.completed_lines / (float)data.image_height)*100.00;
        std::cerr << "[" <<int(percentage_completed) << "%] completed" << std::endl;
    }
}

int main() {
    RenderData render_data; 

    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int max_depth = 25;

    render_data.image_width = image_width;
    render_data.image_height = image_height;
    render_data.samples_per_pixel = samples_per_pixel;
    render_data.max_depth = max_depth;
    render_data.buffer = std::vector<color>(image_width * image_height);

    // Set up Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.02;

    Camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    // Simple usage of creating a Scene
    RTCDevice device = initializeDevice();
    auto cs = make_shared<Scene>(device, cam);

    setup_benchmark_scene(cs, device);

    // When scene construction is finished, the device is no longer needed.
    rtcReleaseDevice(device);

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

    render_data.completed_lines = 0;

    for (int i=0; i < num_threads; i++) {
        // In the first thead, we want the first lines_per_thread lines to be rendered
        threads.emplace_back(render_scanlines_sse,lines_per_thread,(image_height-1) - (i * lines_per_thread), cs, std::ref(render_data),cam);
    }
    threads.emplace_back(render_scanlines_sse,leftOver,(image_height-1) - (num_threads * lines_per_thread), cs, std::ref(render_data),cam);

    for (auto &thread : threads) {
            thread.join();
    }
    std::cerr << "Joining all threads" << std::endl;
    threads.clear();
    std::cout << "P3" << std::endl;
    std::cout << image_width << ' ' << image_height << std::endl;
    std::cout << 255 << std::endl;
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            int buffer_index = j * image_width + i;
            write_color(std::cout, render_data.buffer[buffer_index], samples_per_pixel);
        }
        float percentage_completed = (((float)image_height - (float)j) / (float)image_height)*100.0;
        std::cerr << "[" << (int)percentage_completed << "%] outputting completed" << std::endl;
    }
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
    double time_seconds = elapsed_time / 1000.0;

    std::cerr << "\nCompleted render of scene. Render time: " << time_seconds << " seconds" << "\n";
}