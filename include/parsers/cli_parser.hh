#ifndef CLIPARSER_H
#define CLIPARSER_H

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include "render.h"

/**
 * @struct Config
 * @brief Object to hold all relevant data determined by CLI flags.
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

void outputHelpGuide(std::ostream& out);

void outputRenderInfo(std::ostream& out, Config& config, RenderData& render_data, float time);

int checkValidIntegerInput(int& i, int argc, char* argv[], std::string flagName);

/**
 * @brief given argc, argv, process and return a Config struct containing all the settings.
 * Doesn't account for some invalid input, such as:
 * -> invalid output path
 * -> no checks for if threads or vectorization is supported by hardware
*/
Config parseArguments(int argc, char* argv[]);

#endif
