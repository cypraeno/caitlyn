#ifndef OUTPUT_H
#define OUTPUT_H

#include "render.h"

#include <functional>
#include <fstream>
#include "png_output.h"
#include "cli_parser.hh"

#include "stb_image_write.h"

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
void output(RenderData& render_data, Camera& cam, std::shared_ptr<Scene> scene_ptr, Config& config);

#endif