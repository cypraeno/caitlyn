#include "volume.h"

Volume::Volume(shared_ptr<Medium> medium, shared_ptr<Geometry> geom_ptr, RTCDevice device) : medium{medium}, geom_ptr{geom_ptr}, Geometry(geom_ptr) {
    volume_scene = rtcNewScene(device);
    unsigned int geomID = rtcAttachGeometry(volume_scene, geom_ptr->geom);
    rtcReleaseGeometry(geom_ptr->geom);
    rtcCommitScene(volume_scene);
}

shared_ptr<material> Volume::materialById(unsigned int geomID) const {
    return medium->phase;
}

HitInfo Volume::getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const {
    HitInfo rec = geom_ptr->getHitInfo(r, p, t, geomID);
    rec.medium = true;
    return rec;
}