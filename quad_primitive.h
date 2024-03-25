#ifndef QUAD_PRIMITIVE_H
#define QUAD_PRIMITIVE_H

#include "ray.h"
#include "primitive.h"

struct Vertex3f { float x, y, z; };
struct Quad { int v0, v1, v2, v3; };

class QuadPrimitive : public Primitive {

    private:

        vec3 u, v;
        vec3 normal;
        vec3 w;
        double D;

    public:

        QuadPrimitive(const point3& position, const vec3& _u, const vec3& _v, shared_ptr<material> mat_ptr, RTCDevice device);

        shared_ptr<material> materialById(unsigned int geomID) const override;

        HitInfo getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const override;

        point3 sample(const HitInfo& rec) const override;
        double pdf(const HitInfo& rec, ray sample_ray) const override;
};

#endif
