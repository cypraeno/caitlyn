#include <embree4/rtcore.h>
#include "csr_parser.hh"
#include "cli_parser.hh"
#include "device.h"

#include "output.h"

void brdf_tests() {
    RenderData render_data; 
    const auto aspect_ratio = 16.0 / 9.0;
    setRenderData(render_data, aspect_ratio, 1200, 25, 200);

    point3 lookfrom(10, 3, 0);
    point3 lookat(0, 2, 0);
    vec3 vup(0,1,0);
    double vfov = 60;
    double aperture = 0.0001;
    double dist_to_focus = 10.0;

    Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus);

    RTCDevice device = initializeDevice();
    auto scene_ptr = make_shared<Scene>(device, cam);

    // ALL MATERIALS
    // DEFAULTS V0.1.X
    auto emit = make_shared<emissive>(color(15.0, 15.0, 15.0));
    auto mt0 = make_shared<dielectric>(1.5);
    auto mt1 = make_shared<metal>(color(1.0, 1.0, 1.0), 0.1);
    auto mt2 = make_shared<lambertian>(color(1.0, 1.0, 1.0));

    // Oren-Nayar
    auto mt3 = make_shared<OrenNayar>(color(1.0, 1.0, 1.0), 0.0);

    // Cook-Torrance
    // Complex example:
    auto mt4 = make_shared<CookTorrance>(color(1.0, 1.0, 1.0), color(1.0, 1.0, 1.0), 0.05);
    // Non-complex example:
    auto mt5 = make_shared<CookTorrance>(color(1.0, 1.0, 1.0), 0.1);

    // Dielectric comparison
    auto mt6 = make_shared<CookTorranceDielectric>(color(1.0, 1.0, 1.0), 1.5, 0.0001); // model glass
    auto mt7 = make_shared<CookTorranceDielectric>(color(1.0, 1.0, 1.0), 0.0, 0.0001); // model mirror

    // Example of MixtureBSDF
    std::vector<float> weights = {0.333f, 0.334f, 0.333f};
    std::vector<std::shared_ptr<material>> mats;
    mats.push_back(mt3);
    mats.push_back(mt6);
    mats.push_back(mt7);
    auto mt8 = make_shared<MixtureBSDF>(weights, mats);

    // Adding 3 spheres
    auto sphere1 = make_shared<SpherePrimitive>(point3(0, 2, 2), mt5, 2, device);
    auto sphere2 = make_shared<SpherePrimitive>(point3(1, 2, -2), emit, 0.5, device);
    auto sphere3 = make_shared<SpherePrimitive>(point3(-4, 2, -1), mt8, 2, device);
    scene_ptr->add_primitive(sphere2);
    scene_ptr->add_primitive(sphere3);
    scene_ptr->add_physical_light(sphere2);

    // Create volume out of sphere1
    auto iso = make_shared<isotropic>(color(1,1,1));
    auto medium = make_shared<Medium>(1, iso);
    auto volume1 = make_shared<Volume>(medium, sphere1, device);
    scene_ptr->add_volume(volume1);

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
    CSRParser parser;
    auto scene_ptr = parser.parseCSR(filePath, device);
    scene_ptr->commitScene();
    rtcReleaseDevice(device);

    output(render_data, scene_ptr->cam, scene_ptr, config);
}
