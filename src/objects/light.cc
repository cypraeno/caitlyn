#include "light.h"

emissive::emissive(color emission_color) : emission_color{emission_color} {}

bool emissive::scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const {
    return false;
}

color emissive::emitted(double u, double v, const point3& p) const {
    return emission_color;
}

Light::Light(vec3 position) : Visual(position) {}