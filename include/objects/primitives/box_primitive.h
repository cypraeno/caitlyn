#ifndef BOX_PRIMITIVE_H
#define BOX_PRIMITIVE_H

#include "primitive.h"

/**
 * @class Box
*/
class BoxPrimitive : public Primitive {
    private:
    vec3 a;
    vec3 b;
    vec3 c;

    public:
    BoxPrimitive(const point3& position, const vec3& a, const vec3& b, const vec3& c, std::shared_ptr<material> mat_ptr, RTCDevice device);

    shared_ptr<material> materialById(unsigned int geomID) const override;

    HitInfo getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const;

    vec3 getA();
    vec3 getB();
    vec3 getC();
};

#endif