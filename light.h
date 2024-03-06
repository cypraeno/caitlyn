#ifndef LIGHT_H
#define LIGHT_H

#include "material.h"
#include "visual.h"
#include "ray.h"

/** @brief an emissive material is just one that does not scatter, and adds light. But this is not done from the material. */
class emissive : public material {
    public:
    color emission_color;
    emissive(color emission_color) : emission_color{emission_color} {}
    bool scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const override {
        return false;
    }
    color emitted(double u, double v, const point3& p) const override {
        return emission_color;
    }
};

/**
 * @brief Empty implementation of Light as a Visual. Update as light sampling
 * is implemented and non-physical lights become possible.
*/
class Light : public Visual {
    public:
    Light(vec3 position) : Visual(position) {}
};


#endif