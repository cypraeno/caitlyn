#ifndef LIGHT_H
#define LIGHT_H

#include "material.h"
#include "visual.h"

/** @brief an emissive material is just one that does not scatter, and adds light. But this is not done from the material. */
class emissive : public material {
    public:
    color emission_color;
    emissive(color emission_color) : emission_color{emission_color} {}
    bool scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const override {
        return false;
    }
    color emit() const override {
        return emission_color;
    }
};

class Light : public Visual {
    public:
    Light(vec3 position) : Visual(position) {}
};


#endif