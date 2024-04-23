#include <embree4/rtcore.h>
#include "CSRParser.h"
#include "csr_validator.hh"
#include "device.h"

#include "CLIParser.h"

#include "output.h"

void brdf_tests() {
    RenderData render_data; 
    const auto aspect_ratio = 16.0 / 9.0;
    setRenderData(render_data, aspect_ratio, 1200, 50, 20);

    point3 lookfrom(10, 3, 0);
    point3 lookat(0, 2, 0);
    vec3 vup(0,1,0);
    double vfov = 60;
    double aperture = 0.0001;
    double dist_to_focus = 10.0;

    Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    RTCDevice device = initializeDevice();
    auto scene_ptr = make_shared<Scene>(device, cam);

    //auto mt = make_shared<metal>(color(0.7, 0.6, 0.77), 0.1);
    color test = color(1.0, 1.0, 1.0);
    auto mt = make_shared<lambertian>(test);
    auto mt2 = make_shared<OrenNayar>(test, 0.0);
    
    // Complex example:
    auto mt3 = make_shared<CookTorrance>(color(1.0, 1.0, 1.0), color(1.0, 1.0, 1.0), 0.05);
    // Non-complex example:
    auto mt4 = make_shared<CookTorrance>(color(1.0, 1.0, 1.0), 0.0);

    // Dielectric comparison
    auto mt5 = make_shared<CookTorranceDielectric>(color(0.6, 1.0, 0.6), 1.5, 0.0001); // model glass
    auto mt7 = make_shared<CookTorranceDielectric>(color(0.6, 1.0, 0.6), 0.0, 0.0001); // model mirror

    auto sphere1 = make_shared<SpherePrimitive>(point3(0, 2, 2), mt5, 2, device);
    auto sphere2 = make_shared<SpherePrimitive>(point3(0, 2, -2), mt7, 2, device);
    scene_ptr->add_primitive(sphere1);
    scene_ptr->add_primitive(sphere2);

    auto red     = make_shared<lambertian>(color(1.0, 0.2, 0.2));
    auto ground = make_shared<SpherePrimitive>(point3(0,-10000,0), red, 10000, device);
    scene_ptr->add_primitive(ground);

    scene_ptr->commitScene();
    rtcReleaseDevice(device);

    Config config;
    output(render_data, cam, scene_ptr, config);
}

int main(int argc, char* argv[]) {
    Config config = parseArguments(argc, argv);
    
    RenderData render_data;
    const auto aspect_ratio = static_cast<float>(config.image_width) / config.image_height;
    setRenderData(render_data, aspect_ratio, config.image_width, config.samples_per_pixel, config.max_depth);
    std::string filePath = config.inputFile;
    RTCDevice device = initializeDevice();
    isCSR(filePath);
    CSRParser parser;
    auto scene_ptr = parser.parseCSR(filePath, device);
    scene_ptr->commitScene();
    rtcReleaseDevice(device);

    output(render_data, scene_ptr->cam, scene_ptr, config);
}
