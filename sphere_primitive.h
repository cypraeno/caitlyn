#ifndef SPHERE_PRIMITIVE_H
#define SPHERE_PRIMITIVE_H

#include "primitive.h"

// SPHEREPRIMITIVE INTERFACE
// The most basic sphere. Can only have its radius changed and hold a material.

class SpherePrimitive : public Primitive {
    public:
    double radius;

    SpherePrimitive(vec3 position, shared_ptr<material> mat_ptr, double radius, RTCDevice device) 
        : radius{radius}, Primitive(position, mat_ptr, rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT)) {
        float* spherev = (float*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, 4*sizeof(float), 1);
        if (spherev) {
            spherev[0] = position.x();
            spherev[1] = position.y();
            spherev[2] = position.z();
            spherev[3] = radius;
        }
        rtcCommitGeometry(geom);
    }

    shared_ptr<material> materialById(unsigned int geomID) const override {
        return mat_ptr;
    }

    HitInfo getHitInfo(const ray& r, const vec3& p, unsigned int geomID) const override {
        HitInfo record;
        record.pos = p;
        vec3 outward_normal = (p - position) / radius;
        record.normal = outward_normal;
        record.set_face_normal(r, outward_normal);
        return record;
    }
};

#endif