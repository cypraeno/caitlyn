#ifndef SCENE_H
#define SCENE_H

#include <embree4/rtcore.h>
#include <map>
#include "camera.h"
#include "material.h"
#include "primitive.h"

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

struct HitInfo {
    point3 pos;
    vec3 normal;
    bool front_face;

    /** @brief Given a face's outward normal and the initial ray, sets front_face to represent
    if collision hits it from the front or not. */
    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Scene {
    public:
    Camera cam;
    std::map<unsigned int, std::shared_ptr<Geometry>> geom_map;
    RTCScene rtc_scene;

    // Default Constructor
    // requires a device to initialize RTCScene
    Scene(RTCDevice device, Camera cam);
    ~Scene();
    void commitScene();
    void releaseScene();
    unsigned int add_primitive(std::shared_ptr<Primitive> prim);
};

void add_sphere(RTCDevice device, RTCScene scene);
void add_triangle(RTCDevice device, RTCScene scene);


#endif