#ifndef EMISSIVE_H
#define EMISSIVE_H

#include "material.h"
#include "visual.h"

/** @brief an emissive material is just one that does not scatter, and adds light. But this is not done from the material. */
class emissive : public material {
    bool scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const override {
        return false;
    }
}

class Light : public Visual {
    public:
    Light(vec3 position) : Visual(position) {}
};


#endif