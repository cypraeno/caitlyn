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
        rtcSetGeometryBuildQuality(geom, RTC_BUILD_QUALITY_HIGH);
        rtcCommitGeometry(geom);
    }

    shared_ptr<material> materialById(unsigned int geomID) const override {
        return mat_ptr;
    }

    HitInfo getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const override {
        HitInfo record;
        record.pos = p;
        record.t = t;
        vec3 outward_normal = (p - position) / radius;
        record.normal = outward_normal;
        record.set_face_normal(r, outward_normal);

        double u, v;
        get_sphere_uv(outward_normal, u, v);
        record.u = u;
        record.v = v;

        return record;
    }

    private:
    static void get_sphere_uv(const point3& p, double& u, double& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + pi;

        u = phi / (2*pi);
        v = theta / pi;
    }
};

#endif