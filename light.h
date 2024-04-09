#ifndef LIGHT_H
#define LIGHT_H

#include "material.h"
#include "visual.h"
#include "ray.h"

/** @brief an emissive material is just one that does not scatter, and adds light. But this is not done from the material. */
class emissive : public material {
    public:
    emissive(shared_ptr<texture> a) : emit(a) {}
    emissive(color emission_color) : emit(make_shared<solid_color>(emission_color)) {}
    bool scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const override {
        return false;
    }
    color emitted(double u, double v, const point3& p) const override {
        return emit->value(u, v, p);
    }

    private:
    shared_ptr<texture> emit;
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