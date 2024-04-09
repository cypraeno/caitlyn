#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include "render.h"

/**
 * @brief structure to hold all relevant data determined by flags.
 * Automatically holds defaults.
 * @note A little redundant with the existence of RenderData. Unsure how to reconcile.
*/
struct Config {

    // Standard flags
    int samples_per_pixel = 50;
    int max_depth = 50;
    std::string inputFile = "scene.csr";
    std::string outputPath = "image.ppm";
    int image_width = 1200;
    int image_height = 675;
    std::string outputType = "ppm"; // [jpg|png|ppm]
    
    // Output flags
    bool showVersion = false;
    bool showHelp = false;
    bool verbose = false;
    std::string debugFile = "debug.txt";

    // Optimization flags
    bool multithreading = false;
    int threads = -1; // if -1, then uses hardware concurrency. only used if multithreading is true.
    int vectorization = 0; // [NONE|4|8|16], NONE = 0    

};

void outputHelpGuide(std::ostream& out) {
    out << "Usage: ./caitlyn [options]\n"
        << "Options:\n"
        << " -s,  --samples <number>               Number of samples per pixel.\n"
        << " -d,  --depth <number>                 Maximum recursion depth for rays.\n"
        << " -i,  --input <filepath>               Input file path for the scene.\n"
        << " -o,  --output <path>                  Output path for the rendered image.\n"
        << " -r,  --resolution <width> <height>    Resolution of the output image.\n"
        << " -t,  --type <image_type>              Type of the output image [png|jpg|ppm].\n"
        << " -mt, --multithreading                 Enable multithreading.\n"
        << " -v,  --version                        Show the current version.\n"
        << " -h,  --help                           Show this help message.\n"
        << " -V,  --verbose                        Enables more descriptive messages of scenes and rendering process.\n"
        << " -T,  --threads <amt>                  If multithreading is enabled, sets amount of threads used.\n"
        << " -Vx, --vectorization <batch_size>     Set SIMD vectorization batch size [0|4|8|16]. If NONE = 0, do not enable the flag.\n";
    exit(0);
}

void outputRenderInfo(std::ofstream& out, Config& config, RenderData& render_data, float time) {
    out << "======== " << config.inputFile << " ========" << std::endl;
    out << "Samples: " << render_data.samples_per_pixel << std::endl;
    out << "Depth: " << render_data.max_depth << std::endl;
    out << "Time: " << time << " seconds" << std::endl;
    if (config.multithreading) { out << "Multithreading: YES" << std::endl; }
    else { out << "Multithreading: NO" << std::endl; }
    if (config.vectorization == 0) { out << "Vectorization: NONE" << std::endl; }
    else{ out << "Vectorization: " << config.vectorization << std::endl; }
}

int checkValidIntegerInput(int& i, int argc, char* argv[], std::string flagName) {
    int result;
    if(i + 1 < argc) { // Make sure we aren't at the end of argv
        try {
            result = std::stoi(argv[++i]); // Increment 'i' and get the next argument
            if (result <= 0) {
                throw std::invalid_argument("Input must be a positive integer.");
            }
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument("Invalid argument for "+flagName+": Argument must be a positive integer.");
        } catch (const std::out_of_range& e) {
            throw std::out_of_range("Invalid argument for "+flagName+": Argument is out of range.");
        }
    } else {
        throw std::invalid_argument("Missing argument for "+flagName+".");
    }
    return result;
}

/**
 * @brief given argc, argv, process and return a Config struct containing all the settings.
 * Doesn't account for some invalid input, such as:
 * -> invalid output path
 * -> no checks for if threads or vectorization is supported by hardware
*/
Config parseArguments(int argc, char* argv[]) {
    Config config;
    for(int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if(arg == "-s" || arg == "--samples") {
            config.samples_per_pixel = checkValidIntegerInput(i, argc, argv, "-s/--samples");
        } else if(arg == "-d" || arg == "--depth") {
            config.max_depth = checkValidIntegerInput(i, argc, argv, "-d/--depth");
        } else if(arg == "-i" || arg == "--input") {
            if(i + 1 < argc) {
                config.inputFile = argv[++i];
            }
        } else if(arg == "-o" || arg == "--output") {
            if(i + 1 < argc) {
                config.outputPath = argv[++i];
            }
        } else if(arg == "-r" || arg == "--resolution") {
            config.image_width = checkValidIntegerInput(i, argc, argv, "-r/--resolution <width> <height>");
            config.image_height = checkValidIntegerInput(i, argc, argv, "-r/--resolution <width> <height>");
        } else if(arg == "-t" || arg == "--type") {
            if(i + 1 < argc) {
                std::string type(argv[++i]);
                if (type == "ppm" || type == "png" || type == "jpg") {
                    config.outputType = type;
                } else {
                    throw std::invalid_argument("Invalid argument for -t/--type [ppm]");
                }
            }
        } else if(arg == "-m" || arg == "--multithreading") {
            config.multithreading = true;
        } else if(arg == "-T" || arg == "--threads") {
            config.threads = checkValidIntegerInput(i, argc, argv, "-T/--threads");
        } else if(arg == "-Vx" || arg == "--vectorization") {
            int choice = checkValidIntegerInput(i, argc, argv, "-Vx/--vectorization");
            if (choice == 0 || choice == 4 || choice == 8 || choice == 16) {
                config.vectorization = choice;
            } else {
                throw std::invalid_argument("Error: Invalid option for --vectorization [1|4|8|16]. Use '--help' for more information.");
            }
        } else if(arg == "-v" || arg == "--version") {
            config.showVersion = true;
            std::cout << "caitlyn version 0.1.2" << std::endl;
        } else if(arg == "-h" || arg == "--help") {
            config.showHelp = true;
            outputHelpGuide(std::cout);
        } else if(arg == "-V" || arg == "--verbose") {
            config.verbose = true;
        }
        if (i + 1 < argc && argv[i + 1][0] != '-') {
            throw std::invalid_argument("Too many arguments for "+arg+" flag.");
        }
    }
    return config;
}