#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "visual.h"
#include "material.h"
#include "hitinfo.h"
#include <embree4/rtcore.h>

class Geometry : public Visual {
    public:
    Geometry(vec3 position) : Visual(position) {}

    /** @brief given a geomID (referring to ID given by scene attachment), find the material pointer. Usually called by renderer. */
    virtual shared_ptr<material> materialById(unsigned int geomID) const = 0;

    virtual HitInfo getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const = 0;

    virtual point3 sample(const HitInfo& rec) const {
        return point3(0,0,0);
    };
    virtual double pdf(const HitInfo& rec, ray sample_ray) const {
        return 0.0;
    };
};

#endif