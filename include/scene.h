#ifndef SCENE_H
#define SCENE_H

#include <embree4/rtcore.h>
#include <map>
#include "camera.h"
#include "material.h"
#include "light.h"
#include "sphere_primitive.h"
#include "instances.h"
#include "hitinfo.h"
#include "volume.h"

// SCENE INTERFACE
// The scene class object covers all relevant objects in a scene:
// => Camera
// => Meshes
// => Lights

// RAYTRACING
// => When a ray is cast on a scene, rtcIntersect1 returns a geomID.
// => When objects are added, their geomIDs are mapped to a material.
// => Thus, a geomID tells renderer how to scatter the ray.

// METHODS
// => Scene can have emissives and meshes added to it.
// => Once everything is added, the user commits the scene.

class Scene {
    public:
    Camera cam;
    std::map<unsigned int, std::shared_ptr<Geometry>> geom_map;
    RTCScene rtc_scene;

    std::vector<std::shared_ptr<Geometry>> physical_lights;
    std::vector<std::shared_ptr<Light>> lights;

    // Default Constructor
    // requires a device to initialize RTCScene
    Scene(RTCDevice device, Camera cam);
    ~Scene();
    void commitScene();
    void releaseScene();
    unsigned int add_primitive(std::shared_ptr<Primitive> prim);
    unsigned int add_volume(std::shared_ptr<Volume> vol);

    void add_physical_light(std::shared_ptr<Geometry> geom_ptr);
    unsigned int add_primitive_instance(std::shared_ptr<PrimitiveInstance> pi_ptr, RTCDevice device);
};

void add_sphere(RTCDevice device, RTCScene scene);
void add_triangle(RTCDevice device, RTCScene scene);


#endif
