#ifndef VOLUME_H
#define VOLUME_H

#include "geometry.h"
#include "medium.h"

class Volume : public Geometry {
    public:
    // Releases geometry pointer of the geom ptr, and keeps for itself. Thus, the used geometry cannot be added itself.
    Volume(shared_ptr<Medium> medium, shared_ptr<Geometry> geom, RTCDevice device);

    virtual shared_ptr<material> materialById(unsigned int geomID) const override;

    virtual HitInfo getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const override;

    // sample and pdf are not implemented as volumes should not be sampled for direct light sampling
    // however, reconsider with emissive volumes

    RTCScene volume_scene; // object scene only containing the volume
    shared_ptr<Medium> medium;
    shared_ptr<Geometry> geom_ptr;
};

#endif