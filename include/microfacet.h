#ifndef MICROFACET_H
#define MICROFACET_H

#include "general.h"
#include "onb.h"

/**
 * @class MicrofacetDF
 * @brief Classes containing definitions to sample and weight for
 * microfacet distributions in BRDFs.
*/
class Microfacet {
    public:
    float roughness;

    Microfacet(float roughness) : roughness{roughness} {}

    virtual float D(float cosTheta) const = 0;
    virtual float G(float cosTheta_V, float cosTheta_L) const = 0;
    virtual vec3 sample(vec3 normal) const = 0;
};

class GGX : public Microfacet {
    public:
    GGX(float roughness) : Microfacet(roughness) {}

    float D(float NoH) const override {
        float r = fmax(0.0001, roughness);
        // float alpha = r * r;
        // float alpha2 = alpha * alpha;
        float alpha2 = r * r;
        float NoH2 = NoH * NoH;
        float b = (NoH2 * (alpha2 - 1.0) + 1.0);
        // return (alpha2 / pi) / (b * b);
        return 1;
        // - The D term in GGX is known to scale at ridiculous amounts to overflow to inf when multiple products, as seen in layering.
        //   For now, since we know that D exists in the f and pdf, it is safe to arbitrarily set it to 1 or omit it completely, but a better solution is needed.
    }

    float G(float NoV, float NoL) const override {
        return G1_GGX_Schlick(NoL) * G1_GGX_Schlick(NoV);
    }

    vec3 sample(vec3 N) const override {
        onb ortho;
        ortho.build_from_w(N);

        auto e1 = random_double();
        auto e2 = random_double();

        float theta = atan(roughness * sqrt(e1 / (1 - e1)));
        float phi = 2 * pi * e2;

        // Calculate normal with y as the up vector
        float x = sin(theta) * cos(phi);
        float y = sin(theta) * sin(phi);
        float z = cos(theta);
        vec3 microfacet_normal = vec3(x, y, z).unit_vector();

        vec3 adjusted = ortho.local(microfacet_normal);
        return adjusted;
    }

    private:
    float G1_GGX_Schlick(float NoV) const {
        float alpha = roughness * roughness;
        float k = alpha / 2.0;
        return fmax(NoV, 0.001) / (NoV * (1.0 - k) + k);
    }
};

#endif