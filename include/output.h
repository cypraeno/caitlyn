#ifndef OUTPUT_H
#define OUTPUT_H

#include "render.h"

#include <functional>
#include <fstream>
#include "png_output.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

// Threading
#include <vector>
#include <thread>

struct RGB{
    unsigned char R;
    unsigned char G;
    unsigned char B;
};

/**
 * @brief modified version of other output that takes in CLI arguments and modifies behaviour.
 * @note eventually should REPLACE the other one. The other one exists to keep other scenes intact.
 * @note In the future, RenderData can be replaced by Config (or the other way around).
 * The render functions should be modified to take in a std::vector<color> buffer(image_width * image_height);
 * since that is the only property of RenderData needed that DOES NOT EXIST in Config.
*/
void output(RenderData& render_data, Camera& cam, std::shared_ptr<Scene> scene_ptr, Config& config) {
    int image_height = render_data.image_height;
    int image_width = render_data.image_width;
    int samples_per_pixel = render_data.samples_per_pixel;

    auto start_time = std::chrono::high_resolution_clock::now();
    render_data.completed_lines = 0;

    std::function<void(int, int, std::shared_ptr<Scene>, RenderData&, Camera)> render_function;

    if (config.vectorization == 0) { render_function = render_scanlines; }
    else if (config.vectorization == 4) { render_function = render_scanlines_sse; }
    else if (config.vectorization == 8) { render_function = render_scanlines_avx; }
    else if (config.vectorization == 16) { render_function = render_scanlines_avx; } // replace with 16 batch render_scanlines when made
    else { render_function = render_scanlines; }
    if (!config.multithreading) {
        render_function(image_height, image_height-1, scene_ptr, render_data, cam);
    } else {
        int num_threads;
        if (config.threads == -1) {
            // Threading approach? : Divide the scanlines into N blocks
            num_threads = std::thread::hardware_concurrency() - 1;
        } else {
            num_threads = config.threads;
        }
        // Image height is the number of scanlines, suppose image_height = 800
        const int lines_per_thread = image_height / num_threads;
        const int leftOver = image_height % num_threads;
        // The first <num_threads> threads are dedicated <lines_per_thread> lines, and the last thread is dedicated to <leftOver>

        std::vector<color> pixel_colors;
        std::vector<std::thread> threads;

        for (int i=0; i < num_threads; i++) {
            // In the first thead, we want the first lines_per_thread lines to be rendered
            threads.emplace_back(render_function,lines_per_thread,(image_height-1) - (i * lines_per_thread), scene_ptr, std::ref(render_data),cam);
        }
        threads.emplace_back(render_function,leftOver,(image_height-1) - (num_threads * lines_per_thread), scene_ptr, std::ref(render_data),cam);

        for (auto &thread : threads) {
            thread.join();
        }

        if (config.verbose) {std::cerr << "Joining all threads" << std::endl;}
        threads.clear();
    }
    
    // PPM outputting. No current support for JPG and PNG.
    if (config.outputType == "ppm") {
        std::ofstream outFile(config.outputPath);
        if (!outFile.is_open()) {throw std::runtime_error("Could not open file: " + config.outputPath);}
        outFile << "P3" << std::endl;
        outFile << image_width << ' ' << image_height << std::endl;
        outFile << 255 << std::endl;
        for (int j = image_height - 1; j >= 0; --j) {
            for (int i = 0; i < image_width; ++i) {
                int buffer_index = j * image_width + i;
                write_color(outFile, render_data.buffer[buffer_index], samples_per_pixel);
            }
            float percentage_completed = (((float)image_height - (float)j) / (float)image_height)*100.0;
            if (config.verbose) {
                std::cerr << "[" << (int)percentage_completed << "%] outputting completed" << std::endl;
            }
        }
        outFile.close();
    } else if (config.outputType == "jpg") {
        struct RGB data[image_height][image_width];
        for (int j = image_height - 1 ; j >= 0 ; j-- ) {
            for (int i = 0; i < image_width; i++) {
                int buffer_index = j * image_width + i;
                color pixel_color = color_to_256(render_data.buffer[buffer_index], samples_per_pixel);

                data[image_height - j - 1][i].R = pixel_color.x();
                data[image_height - j - 1][i].G = pixel_color.y();
                data[image_height - j - 1][i].B = pixel_color.z();
            }
        }
        if (config.outputPath == "image.ppm") {
            stbi_write_jpg("image.jpg", image_width, image_height, 3, data, 100);
        } else {
            stbi_write_jpg(config.outputPath.c_str(), image_width, image_height, 3, data, 100);
        }
    } else if (config.outputType == "png") {
        if (config.outputPath == "image.ppm") {
            write_png("image.png", image_width, image_height, samples_per_pixel, render_data.buffer);
        } else {
            write_png(config.outputPath.c_str(), image_width, image_height, samples_per_pixel, render_data.buffer);
        }
    }

    if (config.verbose) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
        double time_seconds = elapsed_time / 1000.0;

        std::ofstream debugFile(config.debugFile, std::ios::app);
        if (!debugFile.is_open()) {throw std::runtime_error("Could not open file: " + config.debugFile);}
        outputRenderInfo(debugFile, config, render_data, time_seconds);

        std::cerr << "\nCompleted render of scene. Render time: " << time_seconds << " seconds" << "\n";
    }
}

#endif