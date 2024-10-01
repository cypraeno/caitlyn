#ifndef SPHERE_PRIMITIVE_H
#define SPHERE_PRIMITIVE_H

#include "primitive.h"

// SPHEREPRIMITIVE INTERFACE
// The most basic sphere. Can only have its radius changed and hold a material.

class SpherePrimitive : public Primitive {
    public:
    double radius;

    SpherePrimitive(vec3 position, shared_ptr<material> mat_ptr, double radius, RTCDevice device);

    shared_ptr<material> materialById(unsigned int geomID) const override;

    HitInfo getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const override;

    point3 sample(const HitInfo& rec) const override;
    double pdf(const HitInfo& rec, ray sample_ray) const override;

    private:
    static void get_sphere_uv(const point3& p, double& u, double& v);
    static vec3 random_to_sphere(double radius, double distance_squared);
};

#endif