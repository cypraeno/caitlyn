#ifndef SCENE_H
#define SCENE_H

#include <embree4/rtcore.h>
#include <map>
#include "camera.h"
#include "material.h"
#include "general.h"

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
// => CaitScene can have emissives and meshes added to it.
// => Once everything is added, the user commits the scene.

class CaitScene {
    public:
    camera cam;
    std::map<unsigned int, std::shared_ptr<material>> mat_map;
    RTCScene rtc_scene;

    // Default Constructor
    // requires a device to initialize RTCScene
    CaitScene(RTCDevice device, camera cam);
    ~CaitScene();
    void commitScene();
};

void add_sphere(RTCDevice device, RTCScene scene);
void add_triangle(RTCDevice device, RTCScene scene);


#endif