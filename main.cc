#include <embree4/rtcore.h>
#include "csr_parser.hh"
#include "cli_parser.hh"
#include "device.h"

#include "output.h"

int main(int argc, char* argv[]) {
    Config config = parseArguments(argc, argv);
    
    RenderData render_data;
    const auto aspect_ratio = static_cast<float>(config.image_width) / config.image_height;
    setRenderData(render_data, aspect_ratio, config.image_width, config.samples_per_pixel, config.max_depth);
    std::string filePath = config.inputFile;
    RTCDevice device = initializeDevice();
    CSRParser parser;
    auto scene_ptr = parser.parseCSR(filePath, device);
    scene_ptr->commitScene();
    rtcReleaseDevice(device);

    output(render_data, scene_ptr->cam, scene_ptr, config);
}
