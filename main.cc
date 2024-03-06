#include <embree4/rtcore.h>
#include "CSRParser.h"
#include "device.h"
#include "general.h"
#include "scene.h"
#include "camera.h"
#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "material.h"

#include "sphere_primitive.h"
#include "quad_primitive.h"
#include "intersects.h"
#include "texture.h"
#include "light.h"


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

        color color_from_emission = mat_ptr->emitted(record.u, record.v, record.pos);
        if (!mat_ptr->scatter(r, record, attenuation, scattered)) {
            return color_from_emission;
        } 

        color color_from_scatter = attenuation * colorize_ray(scattered, scene, depth-1);

        return color_from_emission + color_from_scatter;
    }

    // Sky background (gradient blue-white)
    vec3 unit_direction = r.direction().unit_vector();
    auto t = 0.5*(unit_direction.y() + 1.0);

    return color(0, 0, 0); // temp black for lights
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

void setRenderData(RenderData& render_data, const float aspect_ratio, const int image_width,
    const int samples_per_pixel, const int max_depth) {
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    render_data.image_width = image_width;
    render_data.image_height = image_height;
    render_data.samples_per_pixel = samples_per_pixel;
    render_data.max_depth = max_depth;
    render_data.buffer = std::vector<color>(image_width * image_height);
}

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

void completeRayQueueTask(std::vector<RayQueue>& current, std::vector<color>& temp_buffer,
                        std::vector<color>& full_buffer, std::vector<RayQueue>& queue,
                        int mask[], int i, int current_index) {
    // check if theres even any more to do, if not then break out.
    // this pixel is done so we can update the full buffer.
    full_buffer[current_index] += temp_buffer[current_index];
    if (queue.empty()) {
        mask[i] = 0; // disable this part of the packet from running
    } else {
        // replace finished RayQueue with next
        RayQueue back = queue.back();
        queue.pop_back();
        current[i] = back;
    }
}

/**
 * @brief Calculates colours of the given RenderData's buffer according to the assigned lines of pixels.
 * 
 * @note for SSE 4-RayQueue packets scanline rendering
*/
void render_scanlines_sse(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam) {
    int image_width         = data.image_width;
    int image_height        = data.image_height;
    int samples_per_pixel   = data.samples_per_pixel;
    int max_depth           = data.max_depth;

    std::vector<color> full_buffer(image_width);

    std::vector<RayQueue> queue;
    queue.reserve(image_width);

    std::vector<color> temp_buffer(image_width);
    std::vector<color> attenuation_buffer(image_width);
    std::vector<RayQueue> current(4); // size = 4 only

    int mask[4] = {-1, -1, -1, -1};
    
    for (int j=start_line; j>=start_line - (lines - 1); --j) {
        std::fill(full_buffer.begin(), full_buffer.end(), color(0, 0, 0));
        for (int s=0; s < samples_per_pixel; s++) {
            std::fill(temp_buffer.begin(), temp_buffer.end(), color(0, 0, 0));
            std::fill(attenuation_buffer.begin(), attenuation_buffer.end(), color(0, 0, 0));
            queue.clear();
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
                current[i] = back;
            }

            std::fill(std::begin(mask), std::end(mask), -1);
            while (mask[0] != 0 or mask[1] != 0 or mask[2] != 0 or mask[3] != 0) {
                std::vector<ray> rays;
                for (int i=0; i<(int)current.size(); i++) {
                    rays.push_back(current[i].r);
                }
                setupRayHit4(rayhit, rays);
                rtcIntersect4(mask, scene_ptr->rtc_scene, &rayhit);

                HitInfo record;

                for (int i=0; i<4; i++) {
                    if (mask[i] == 0) { continue; }
                    ray current_ray = current[i].r;
                    int current_index = current[i].index;

                    // process each ray by editing the temp_buffer and updating current queue
                    if (rayhit.hit.geomID[i] != RTC_INVALID_GEOMETRY_ID) { // hit
                        ray scattered;
                        color attenuation;
                        std::shared_ptr<Geometry> geomhit = scene_ptr->geom_map[rayhit.hit.geomID[i]];
                        std::shared_ptr<material> mat_ptr = geomhit->materialById(rayhit.hit.geomID[i]);
                        record = geomhit->getHitInfo(current_ray, current_ray.at(rayhit.ray.tfar[i]), rayhit.ray.tfar[i], rayhit.hit.geomID[i]);
                        
                        color color_from_emission = mat_ptr->emitted(record.u, record.v, record.pos);
                        if (!mat_ptr->scatter(current_ray, record, attenuation, scattered)) {
                            if (current[i].depth == 0) { temp_buffer[current_index] = color_from_emission; }
                            else { temp_buffer[current_index] = temp_buffer[current_index] + (attenuation_buffer[current_index] * color_from_emission); }
                            completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                        } else {
                            if (current[i].depth == 0) {
                                temp_buffer[current_index] = color_from_emission;
                                attenuation_buffer[current_index] = attenuation;
                            }
                            else {
                                temp_buffer[current_index] = temp_buffer[current_index] + (attenuation_buffer[current_index] * color_from_emission);
                                attenuation_buffer[current_index] = attenuation_buffer[current_index] * attenuation;
                            }
                            if (current[i].depth + 1 == max_depth) { // reached max depth, replace with next in queue
                                completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
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
                        if (current[i].depth == 0) { temp_buffer[current_index] = multiplier; }
                        else { temp_buffer[current_index] = temp_buffer[current_index] + (attenuation_buffer[current_index] * multiplier); }
                        completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                    }
                }
            }
        }
        for (int i=0; i<image_width; ++i) {
            int buffer_index = j * image_width + i;
            data.buffer[buffer_index] = color(full_buffer[i].x(), full_buffer[i].y(), full_buffer[i].z());
        }
        data.completed_lines += 1;
        float percentage_completed = ((float)data.completed_lines / (float)data.image_height)*100.00;
        std::cerr << "[" <<int(percentage_completed) << "%] completed" << std::endl;
    }
}

/**
 * @brief Calculates colours of the given RenderData's buffer according to the assigned lines of pixels.
 * 
 * @note for AVX 8-RayQueue packets scanline rendering
*/
void render_scanlines_avx(int lines, int start_line, std::shared_ptr<Scene> scene_ptr, RenderData& data, Camera cam) {
    int image_width         = data.image_width;
    int image_height        = data.image_height;
    int samples_per_pixel   = data.samples_per_pixel;
    int max_depth           = data.max_depth;

    const int PACKET_SIZE = 8;

    std::vector<color> full_buffer(image_width);

    std::vector<RayQueue> queue;
    queue.reserve(image_width);

    std::vector<color> temp_buffer(image_width);
    std::vector<RayQueue> current(PACKET_SIZE); // size = 8 only

    int mask[PACKET_SIZE];
    std::fill(mask, mask+PACKET_SIZE, -1);
    
    for (int j=start_line; j>=start_line - (lines - 1); --j) {
        std::fill(full_buffer.begin(), full_buffer.end(), color(0, 0, 0));
        for (int s=0; s < samples_per_pixel; s++) {
            std::fill(temp_buffer.begin(), temp_buffer.end(), color(0, 0, 0));
            queue.clear();
            for (int i=image_width-1; i>=0; --i) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                RayQueue q = { i, 0, r };
                queue.push_back(q);
            }

            RTCRayHit8 rayhit;

            for (int i=0; i<PACKET_SIZE; i++) {
                RayQueue back = queue.back();
                queue.pop_back();
                current[i] = back;
            }

            std::fill(std::begin(mask), std::end(mask), -1);
            bool maskContainsActive = true;
            while (maskContainsActive) {
                maskContainsActive = false;
                for (int p=0; p<PACKET_SIZE; p++) {
                    if (mask[p] == -1) { maskContainsActive = true; }
                    break;
                }

                std::vector<ray> rays;
                for (int i=0; i<(int)current.size(); i++) {
                    rays.push_back(current[i].r);
                }
                setupRayHit8(rayhit, rays);
                rtcIntersect8(mask, scene_ptr->rtc_scene, &rayhit);

                HitInfo record;

                for (int i=0; i<PACKET_SIZE; i++) {
                    if (mask[i] == 0) { continue; }
                    ray current_ray = current[i].r;
                    int current_index = current[i].index;

                    // process each ray by editing the temp_buffer and updating current queue
                    if (rayhit.hit.geomID[i] != RTC_INVALID_GEOMETRY_ID) { // hit
                        ray scattered;
                        color attenuation;
                        std::shared_ptr<Geometry> geomhit = scene_ptr->geom_map[rayhit.hit.geomID[i]];
                        std::shared_ptr<material> mat_ptr = geomhit->materialById(rayhit.hit.geomID[i]);
                        record = geomhit->getHitInfo(current_ray, current_ray.at(rayhit.ray.tfar[i]), rayhit.ray.tfar[i], rayhit.hit.geomID[i]);
                        if (!mat_ptr->scatter(current_ray, record, attenuation, scattered)) { attenuation = color(0,0,0); }
                        if (current[i].depth == 0) { temp_buffer[current_index] = attenuation; }
                        else { temp_buffer[current_index] = temp_buffer[current_index] * attenuation; }
                        if (current[i].depth + 1 == max_depth) { // reached max depth, replace with next in queue
                            completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                        } else { // not finished depth wise
                            current[i].depth += 1;
                            current[i].r = scattered;
                        }
                    } else { // no hit
                        // Sky background (gradient blue-white)
                        vec3 unit_direction = current_ray.direction().unit_vector();
                        auto t = 0.5*(unit_direction.y() + 1.0);

                        color multiplier = (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0); // lerp formula (1.0-t)*start + t*endval
                        if (current[i].depth == 0) { temp_buffer[current_index] = multiplier; } 
                        else { temp_buffer[current_index] = temp_buffer[current_index] * multiplier; }
                        completeRayQueueTask(current, temp_buffer, full_buffer, queue, mask, i, current_index);
                    }
                }
            }
        }
        for (int i=0; i<image_width; ++i) {
            int buffer_index = j * image_width + i;
            data.buffer[buffer_index] = color(full_buffer[i].x(), full_buffer[i].y(), full_buffer[i].z());
        }
        data.completed_lines += 1;
        float percentage_completed = ((float)data.completed_lines / (float)data.image_height)*100.00;
        std::cerr << "[" <<int(percentage_completed) << "%] completed" << std::endl;
    }
}

void output(RenderData& render_data, Camera& cam, std::shared_ptr<Scene> scene_ptr) {
    int image_height = render_data.image_height;
    int image_width = render_data.image_width;
    int samples_per_pixel = render_data.samples_per_pixel;

    // Start Render Timer 
    auto start_time = std::chrono::high_resolution_clock::now();
    render_data.completed_lines = 0;

    // To render entire thing without multithreading, uncomment this line and comment out num_threads -> threads.clear()
    //render_scanlines_sse(image_height,image_height-1,scene_ptr,render_data,cam);

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
        threads.emplace_back(render_scanlines,lines_per_thread,(image_height-1) - (i * lines_per_thread), scene_ptr, std::ref(render_data),cam);
    }
    threads.emplace_back(render_scanlines,leftOver,(image_height-1) - (num_threads * lines_per_thread), scene_ptr, std::ref(render_data),cam);

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

void random_spheres() {
    RenderData render_data; 

    const auto aspect_ratio = 3.0 / 2.0;
    setRenderData(render_data, aspect_ratio, 1200, 50, 50);

    // Set up Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    Camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    // Simple usage of creating a Scene
    RTCDevice device = initializeDevice();
    auto scene_ptr = make_shared<Scene>(device, cam);

    setup_benchmark_scene(scene_ptr, device);

    // When scene construction is finished, the device is no longer needed.
    rtcReleaseDevice(device);

    output(render_data, cam, scene_ptr);
}

void two_spheres() {
    // Set RenderData
    RenderData render_data; 
    const auto aspect_ratio = 16.0 / 9.0;
    setRenderData(render_data, aspect_ratio, 400, 50, 50);

    // Set up Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.0001;
    
    Camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    // Simple usage of creating a Scene
    RTCDevice device = initializeDevice();
    auto scene_ptr = make_shared<Scene>(device, cam);

    // Set World
    auto checker = make_shared<checker_texture>(0.8, color(.2, .3, .1), color(.9, .9, .9));
    auto checkered_surface = make_shared<lambertian>(checker);
    auto sphere1 = make_shared<SpherePrimitive>(point3(0,-10, 0), checkered_surface, 10, device);
    auto sphere2 = make_shared<SpherePrimitive>(point3(0,10, 0), checkered_surface, 10, device);
    scene_ptr->add_primitive(sphere1);
    scene_ptr->add_primitive(sphere2);

    scene_ptr->commitScene();

    rtcReleaseDevice(device);

    output(render_data, cam, scene_ptr);
}

void earth() {
    // Set RenderData
    RenderData render_data; 
    const auto aspect_ratio = 16.0 / 9.0;
    setRenderData(render_data, aspect_ratio, 400, 50, 50);

    // Set up Camera
    point3 lookfrom(0,0,12);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.0001;
    
    Camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    // Simple usage of creating a Scene
    RTCDevice device = initializeDevice();
    auto scene_ptr = make_shared<Scene>(device, cam);

    // Set World
    auto earth_texture = make_shared<image_texture>("../images/earthmap.jpg");
    auto earth_surface = make_shared<lambertian>(earth_texture);
    auto globe = make_shared<SpherePrimitive>(point3(0,0,0), earth_surface, 2, device);
    unsigned int groundID = scene_ptr->add_primitive(globe);

    scene_ptr->commitScene();

    // When scene construction is finished, the device is no longer needed.
    rtcReleaseDevice(device);

    output(render_data, cam, scene_ptr);
}

/**
 * @brief loads "example.csr" in the same directory.
 * @note see example.csr in cypraeno/csr_schema repository
*/
void load_example() {
    RenderData render_data;
    const auto aspect_ratio = 16.0 / 9.0;
    setRenderData(render_data, aspect_ratio, 400, 50, 50);
    std::string filePath = "example.csr";
    RTCDevice device = initializeDevice();
    CSRParser parser;
    auto scene_ptr = parser.parseCSR(filePath, device);
    scene_ptr->commitScene();
    rtcReleaseDevice(device);

    output(render_data, scene_ptr->cam, scene_ptr);
}

void quads() {
    RenderData render_data; 
    const auto aspect_ratio = 16.0 / 9.0;
    setRenderData(render_data, aspect_ratio, 400, 100, 50);

    // Set up Camera
    point3 lookfrom(0, 0, 9);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    double vfov = 80;
    double aperture = 0.0001;
    double dist_to_focus = 10.0;

    Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    // Simple usage of creating a Scene
    RTCDevice device = initializeDevice();
    auto scene_ptr = make_shared<Scene>(device, cam);

    // Materials
    auto left_red     = make_shared<lambertian>(color(1.0, 0.2, 0.2));
    auto back_green   = make_shared<lambertian>(color(0.2, 1.0, 0.2));
    auto right_blue   = make_shared<lambertian>(color(0.2, 0.2, 1.0));
    auto upper_orange = make_shared<lambertian>(color(1.0, 0.5, 0.0));
    auto lower_teal   = make_shared<lambertian>(color(0.2, 0.8, 0.8));

    // Quads
    auto quad1 = make_shared<QuadPrimitive>(point3(-3,-2, 5), vec3(0, 0,-4), vec3(0, 4, 0), left_red, device);
    auto quad2 = make_shared<QuadPrimitive>(point3(-2,-2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_green, device);
    auto quad3 = make_shared<QuadPrimitive>(point3( 3,-2, 1), vec3(0, 0, 4), vec3(0, 4, 0), right_blue, device);
    auto quad4 = make_shared<QuadPrimitive>(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_orange, device);
    auto quad5 = make_shared<QuadPrimitive>(point3(-2,-3, 5), vec3(4, 0, 0), vec3(0, 0,-4), lower_teal, device);

    scene_ptr->add_primitive(quad1);
    scene_ptr->add_primitive(quad2);
    scene_ptr->add_primitive(quad3);
    scene_ptr->add_primitive(quad4);
    scene_ptr->add_primitive(quad5);
    
    scene_ptr->commitScene();

    rtcReleaseDevice(device);

    output(render_data, cam, scene_ptr);
}


void simple_light() {
    RenderData render_data; 
    const auto aspect_ratio = 16.0 / 9.0;
    setRenderData(render_data, aspect_ratio, 400, 100, 50);

    // Set up Camera
    point3 lookfrom(26,3,6);
    point3 lookat(0,2,0);
    vec3 vup(0,1,0);
    double vfov = 20;
    double aperture = 0.0001;
    double dist_to_focus = 10.0;

    Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    // Simple usage of creating a Scene
    RTCDevice device = initializeDevice();
    auto scene_ptr = make_shared<Scene>(device, cam);
    
    // Materials
    auto red     = make_shared<lambertian>(color(1.0, 0.2, 0.2)); // replace with noise once implemented
    auto green   = make_shared<lambertian>(color(0.2, 1.0, 0.2)); // replace with noise once implemented

    auto sphere1 = make_shared<SpherePrimitive>(point3(0,-1000,0), red, 1000, device);
    auto sphere2 = make_shared<SpherePrimitive>(point3(0,2,0), green, 2, device);

    auto lightmaterial = make_shared<emissive>(color(6,6,6));
    auto lightsphere = make_shared<SpherePrimitive>(point3(0,7,0), lightmaterial, 2, device);
    auto lightquad = make_shared<QuadPrimitive>(point3(3,1,-2), vec3(2,0,0), vec3(0,2,0), lightmaterial, device);

    // Add to Scene
    scene_ptr->add_primitive(lightquad);
    scene_ptr->add_primitive(lightsphere);
    scene_ptr->add_primitive(sphere1);
    scene_ptr->add_primitive(sphere2);

    scene_ptr->commitScene();

    rtcReleaseDevice(device);

    output(render_data, cam, scene_ptr);
}

void cornell_box() {
    RenderData render_data; 
    const auto aspect_ratio = 1.0;
    setRenderData(render_data, aspect_ratio, 600, 20, 20);

    // Set up Camera
    point3 lookfrom(278, 278, -800);
    point3 lookat(278, 278, 0);
    vec3 vup(0,1,0);
    double vfov = 40;
    double aperture = 0.0001;
    double dist_to_focus = 10.0;

    Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    // Simple usage of creating a Scene
    RTCDevice device = initializeDevice();
    auto scene_ptr = make_shared<Scene>(device, cam);

    // Materials
    auto red   = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto lightmaterial = make_shared<emissive>(color(15,15,15));

    auto quad1 = make_shared<QuadPrimitive>(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green, device);
    auto quad2 = make_shared<QuadPrimitive>(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), red, device);
    auto quad3 = make_shared<QuadPrimitive>(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), lightmaterial, device);
    auto quad4 = make_shared<QuadPrimitive>(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), white, device);
    auto quad5 = make_shared<QuadPrimitive>(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white, device);
    auto quad6 = make_shared<QuadPrimitive>(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), white, device);

    // Add to Scene
    scene_ptr->add_primitive(quad1);
    scene_ptr->add_primitive(quad2);
    scene_ptr->add_primitive(quad3);
    scene_ptr->add_primitive(quad4);
    scene_ptr->add_primitive(quad5);
    scene_ptr->add_primitive(quad6);

    scene_ptr->commitScene();

    rtcReleaseDevice(device);

    output(render_data, cam, scene_ptr);
}

int main() {
    switch (7) {
        case 1:  random_spheres(); break;
        case 2:  two_spheres();    break;
        case 3:  earth();          break;
        case 4:  quads();          break;
        case 5:  load_example();   break;
        case 6:  simple_light();   break;
        case 7:  cornell_box();    break;
    }
}

