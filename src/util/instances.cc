#include "instances.h"

Instance::Instance(float* transform) : transform{transform} {}

PrimitiveInstance::PrimitiveInstance(float* transform) : Instance(transform) {}

SpherePrimitiveInstance::SpherePrimitiveInstance(std::shared_ptr<SpherePrimitive> sprim, float* transform, RTCDevice device) : PrimitiveInstance(transform) {
    instance_scene = rtcNewScene(device);
    unsigned int geomID = rtcAttachGeometry(instance_scene, sprim->geom);
    rtcReleaseGeometry(sprim->geom);
    rtcCommitScene(instance_scene);

    vec3 translate = vec3(transform[3], transform[7], transform[11]);
    pptr = make_shared<SpherePrimitive>(sprim->position + translate, sprim->mat_ptr, 1, device);
}

QuadPrimitiveInstance::QuadPrimitiveInstance(std::shared_ptr<QuadPrimitive> sprim, float* transform, RTCDevice device) : PrimitiveInstance(transform) {
    instance_scene = rtcNewScene(device);
    unsigned int geomID = rtcAttachGeometry(instance_scene, sprim->geom);
    rtcReleaseGeometry(sprim->geom);
    rtcCommitScene(instance_scene);

    vec3 translate = vec3(transform[3], transform[7], transform[11]);
    vec3 u = sprim->getU();
    vec3 v = sprim->getV();
    pptr = make_shared<QuadPrimitive>(sprim->position + translate, u, v, sprim->mat_ptr, device);
}

BoxPrimitiveInstance::BoxPrimitiveInstance(std::shared_ptr<BoxPrimitive> sprim, float* transform, RTCDevice device) : PrimitiveInstance(transform) {
    instance_scene = rtcNewScene(device);
    unsigned int geomID = rtcAttachGeometry(instance_scene, sprim->geom);
    rtcReleaseGeometry(sprim->geom);
    rtcCommitScene(instance_scene);

    vec3 translate = vec3(transform[3], transform[7], transform[11]);
    vec3 a = sprim->getA();
    vec3 b = sprim->getB();
    vec3 c = sprim->getC();
    pptr = make_shared<BoxPrimitive>(sprim->position + translate, a, b, c, sprim->mat_ptr, device);
}
