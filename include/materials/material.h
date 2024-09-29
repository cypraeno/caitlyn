#ifndef MATERIAL_H
#define MATERIAL_H

#include "general.h"
#include "hit_info.hh"
#include "texture.h"
#include "onb.h"

#include <complex>
#include "microfacet.h"

class hit_record;
class Medium;

// CONSTANTS
const float SPECULAR_ROUGHNESS_SAMPLING_CUTOFF = 0.1;

enum BSDF_TYPE {
    DIFFUSE,
    GLOSSY,
    SPECULAR,
    TRANSMISSION,
    TRANSPARENT
};
struct BSDFSample {

    bool scatter;
    vec3 scatter_direction;
    color bsdf_value;
    float pdf_value;
    BSDF_TYPE type = BSDF_TYPE::DIFFUSE;
};

class material {

    public:
        virtual color emitted(double u, double v, const point3& p) const;
        virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const;
        virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const;
	    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const;

        virtual BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const;
};

/**
 * @class lambertian
 * @brief Implements basic lambertian material with cosine direction sampling.
 * @deprecated Use Oren-Nayar for diffuse if possible. At some point, CSR schema should use Diffuse and defualt to Oren-Nayar anyways.
*/
class lambertian : public material {

    public:
        lambertian(const color& a) : albedo(make_shared<solid_color>(a)) {}
        lambertian(shared_ptr<texture> a) : albedo(a) {}

        virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const override {
            onb uvw;
            uvw.build_from_w(rec.normal);
            auto scatter_direction = uvw.local(random_cosine_direction());
            scattered = ray(rec.pos, scatter_direction, r_in.time());
            
            return true;
        }

        // A lambertians BRDF value is its albedo / pi
        virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
            return albedo->value(rec.u, rec.v, rec.pos) / pi;
        }
        
        virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
            auto cos_theta = dot(rec.normal, scattered.direction().unit_vector());
            return fmax(0.0, cos_theta / pi);
        }

    private:
    shared_ptr<texture> albedo;
};


/**
 * @class metal
 * @brief Implements simple coloured perfect specular with fuzz, which is done by randomly warping output direction. Is not physically accurate
 * since fuzz is not taken into account in f or pdf. Fresnel is not used.
 * 
 * @deprecated Use CookTorrance instead. CSR should at some point use Metal, which is CookTorrance and NOT this class.
 * 
*/
class metal : public material {

    public:

        metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const override {
            vec3 reflected = reflect(r_in.direction().unit_vector(), rec.normal);
            scattered = ray(rec.pos, reflected + fuzz*random_in_unit_sphere(), r_in.time());

            return (dot(scattered.direction(), rec.normal) > 0);
        }

        virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
            return albedo;
        }

        virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
            return 1.0;
        }

        virtual BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const override {
            BSDFSample sample_data;
            // Sample the microfacet distribution to get the scatter direction.
            color attenuation; // placeholder until it gets removed from the scatter function header
            sample_data.scatter = scatter(r_in, rec, attenuation, scattered);
            sample_data.scatter_direction = scattered.direction().unit_vector();

            // Sample the BRDF for the value
            sample_data.bsdf_value = generate(r_in, scattered, rec);

            // Find the PDF for the MDF
            sample_data.pdf_value = pdf(r_in, scattered, rec);

            // Provide type for sample
            if (fuzz < SPECULAR_ROUGHNESS_SAMPLING_CUTOFF) { sample_data.type = BSDF_TYPE::SPECULAR; } // 0.05 was picked arbitrarily, should experiment
            else { sample_data.type = BSDF_TYPE::GLOSSY; }
            return sample_data;
        }

    public:

        color albedo;
        double fuzz;
};

/**
 * @class dielectric
 * @brief Implements simple coloured perfect dielectric (without implementing roughness). Is physically incorrect, does not use fresnel or
 * any worthwhile techniques.
 * 
 * @deprecated Use CookTorranceDielectric instead. CSR should at some point use Transmission, which is CookTorranceDielectric and NOT this class.
 * 
*/
class dielectric : public material {

    public:

        dielectric(double index_of_refraction) : ir(index_of_refraction) {}

        virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const override {
            // If the hit is on the front face, ir is the refracted index.
            // If the hit comes from the outside, then 1.0 is the refracted index (air)
            double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

            vec3 unit_direction = r_in.direction().unit_vector();
            double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            vec3 direction;

            if (refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, refraction_ratio) > random_double()) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, refraction_ratio);
            }
            scattered = ray(rec.pos, direction, r_in.time());
            return true;
        }

        virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
            return color(1.0, 1.0, 1.0);
        }

        virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
            return 1.0;
        }

        virtual BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const override {
            BSDFSample sample_data;
            // Sample the microfacet distribution to get the scatter direction.
            color attenuation; // placeholder until it gets removed from the scatter function header
            sample_data.scatter = scatter(r_in, rec, attenuation, scattered);
            sample_data.scatter_direction = scattered.direction().unit_vector();

            // Sample the BRDF for the value
            sample_data.bsdf_value = generate(r_in, scattered, rec);

            // Find the PDF for the MDF
            sample_data.pdf_value = pdf(r_in, scattered, rec);

            // Provide type for sample
            sample_data.type = BSDF_TYPE::SPECULAR; 
            // is incorrect. only works because render function uses direct light sampling + bsdf sampling the same way for specular and transmission.

            return sample_data;
        }

    public:

        double ir; // Index of Refraction

    private:
    
        // Christophe Schlick's approximation (probability of reflectance)
        // https://en.wikipedia.org/wiki/Schlick%27s_approximation
        static double reflectance(double cosine, double ref_idx) {
            auto r0 = (1 - ref_idx) / (1 + ref_idx);
            r0 = r0 * r0;

            return r0 + (1 - r0) * pow((1 - cosine), 5);
        }
};

/**
 * @class OrenNayar
 * @brief Implements the Oren-Nayar reflectance model for simulating the appearance of rough diffuse surfaces.
 * The Oren-Nayar reflectance model is an extension of the Lambertian model that accounts for the roughness of the
 * surface, providing a more accurate representation of diffuse reflection from surfaces that are not perfectly smooth.
 * 
 * @note The correctness of the pdf and scatter functions, which use cosine-weighted sampling similar to the Lambertian
 * class, may need further verification.
 * 
*/
class OrenNayar : public material {

    public:
    OrenNayar(color albedo, float roughness) : albedo{albedo}, roughness{roughness} {}

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const override;

    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const override;

    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const override;

    private:
    color albedo;
    float roughness;
};

/**
 * @class CookTorrance
 * @brief Implements the Cook-Torrance BRDF model for simulating the specular reflection of a conductor.
 * This is a more complex model of the original `metal` material. 
 * This implementation uses the GGX (Trowbridge-Reitz) microfacet distribution to simulate the roughness.
 * 
 * @param albedo [OPTIONAL] Base colour.
 * @param roughness In range [0-1] defines how rough the surface of the material becomes (less shiny).
 * @param absorption [OPTIONAL] RGB value of how much to not absorb. The higher the color channel, the less that one is absorbed.
 * @param refraction [OPTIONAL] RGB value of how much to refract (i.e NOT reflect). The higher the color channel, the less it shows.
 * 
 * @note by default, if no MDF is specified in the constructor, GGX is used.
*/
class CookTorrance : public material {

    public:
    CookTorrance(color albedo, float roughness)
        : complex{false}, albedo{albedo}, MDF{std::make_shared<GGX>(roughness)} {}

    CookTorrance(color albedo, std::shared_ptr<Microfacet> mdf)
        : complex(false), albedo{albedo}, MDF{mdf} {}

    CookTorrance(color absorption, color refraction, float roughness)
        : complex(true), absorption_coefficient{absorption}, eta{refraction}, MDF{std::make_shared<GGX>(roughness)} {}

    CookTorrance(color absorption, color refraction, std::shared_ptr<Microfacet> mdf)
        : complex(true), absorption_coefficient{absorption}, eta{refraction}, MDF{mdf} {}

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const override;
    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const override;
    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const override;
    virtual BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const override;

    private:
    bool complex;
    color albedo;
    color absorption_coefficient;
    color eta;
    std::shared_ptr<Microfacet> MDF;

    // F, G, D functions
    vec3 fresnelSchlick(float cosTheta, vec3 F0) const;
    float FrComplex(float cosTheta_i, std::complex<float> eta) const;
    vec3 FrComplex(float cosTheta_v, vec3 k, vec3 eta) const;

};

/**
 * @class CookTorranceDielectric
 * @brief Implements the Cook-Torrance Dielectric BxDF model.
 * This implementation uses the GGX (Trowbridge-Reitz) microfacet distribution to simulate the roughness.
 * 
 * @param albedo
 * @param eta Index of refraction (e.g 1.5 for glass)
 * @param roughness In range [0-1] defines how rough the surface of the material becomes (less shiny).
 * @param complexFresnel Indicates type of F term to calculate. 0 uses FrComplex, and any positive integer is used as the exponent to the
 * Schlick approximation.
 * @note by default, if no MDF is specified in the constructor, GGX is used.
*/
class CookTorranceDielectric : public material {

    public:
    CookTorranceDielectric(color albedo, float eta, float roughness, int complexFresnel = 0) 
        : albedo{albedo}, eta{(eta == 0.0f) ? 0.0f : (float)fmax(eta, 1.0001f)}, 
        MDF{std::make_shared<GGX>(roughness)}, complexFresnel{(int)fmax(complexFresnel, 0)} {}

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const;
    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const;
    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const;
    BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const override;

    //private:
    color albedo;
    float eta;
    std::shared_ptr<Microfacet> MDF;
    int complexFresnel;

    float FrDielectric(float cosTheta_i) const;

    float fresnelSchlick(float cosTheta, int exponent) const;

    private:

    color f_r(const ray& r_in, const HitInfo& rec, const ray& scattered, float R) const;
    float pdf_r(const ray& r_in, const HitInfo& rec, const ray& scattered, float R) const;
    float pdf_t(const ray& r_in, const HitInfo& rec, const ray& scattered, float T) const;
    color f_t(const ray& r_in, const HitInfo& rec, const ray& scattered, float T) const;
};

class isotropic : public material {
    public:
    color albedo;

    isotropic(const color& albedo) : albedo{albedo} {}

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const;
    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const;
    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const;
};


class pixel_lambertian : public material {

    public:
        pixel_lambertian(shared_ptr<PixelImageTexture> a) : albedo(a) {}

        virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const override;
        virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const override;
        virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const override;
        virtual BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const;

    private:
    shared_ptr<PixelImageTexture> albedo;
};

/**
 * @class MixtureBSDF
 * @brief A linearly interpolated mixture of N materials, using a vector of weights and materials.
 * The mixture is done by simple randomization, where the weights decide the frequency that a certain mixed
 * material is used. This means that there is no confusing interpolation between outgoing directions or sampling.
 * 
 * @param weights Vector of floats representing weights of each material. Should sum to 1 to retain energy conservation.
 * @param mats Vector of materials.
 * 
 * @note It is assumed that the given vectors are of the same length and that the weights sum to 1.
*/
class MixtureBSDF : public material {
    public:
    MixtureBSDF(std::vector<float> weights, std::vector<std::shared_ptr<material>> mats) : weights{weights}, mats{mats} {}

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const;
    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const; // assumes that scatter has already been called or sample has already been called, and thus rand is already generated.
    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const;
    BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const override; // NOTE: ASSUMES WEIGHTS IS AT LEAST OF SIZE 1 OTHERWISE BEHAVIOUR IS UNDEFINED

    private:
    std::vector<float> weights;
    std::vector<std::shared_ptr<material>> mats;

    // Returns negative if weights vector has nothing.
    int chooseSampleMaterial(float rand) const;
};

/**
 * @class LayeredBSDF
 * @brief A representation of a top BSDF layered over a bottom BSDF with no medium inside. All sampling is done by tracing ray stochastically
 * through the layers and sampling each time.
 * 
 * @param top Top Layer BSDF.
 * @param bottom Bottom Layer BSDF.
 * @param termination Russian Roulette termination condition for number of bounces. If exceeded, pretends light is absorbed.
 * 
 * @note SCATTER, GENERATE, AND PDF ARE NOT READY.
 * USE SAMPLE INSTEAD.
*/
class LayeredBSDF : public material {
    public:
    LayeredBSDF(std::shared_ptr<material> top, std::shared_ptr<material> bottom, std::shared_ptr<Medium> medium, int termination = 10)
        : top{top}, bottom{bottom}, medium{medium}, termination{termination} {}

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const {
        return true;
    }

    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
        return color(1,1,1);
    }

    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
        return 1.0;
    }

    // NOTES:
    // - The D term in GGX is known to scale at ridiculous amounts to overflow to inf when multiple products, as seen in layering.
    //   For now, since we know that D exists in the f and pdf, it is safe to arbitrarily set it to 1 or omit it completely, but a better solution is needed.
    // - Russian Roulette termination does not return black. This is to avoid black specks, but is PHYSICALLY IMPLAUSIBLE.
    //   There may be a better solution!
    //   We can check if the ray is bouncing back towards the INITIAL LAYER by if rec.front_face = on_top
    //   This means that it can never exit via the non-initial layer as a result of Russian Roulette. This is a sacrifice becasue
    //   there are currently no flags to check if a layer is transmissible or not to exit. However, we know that the initial layer must be.
    BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const override;

    private:
    float thickness = 1.0;

    int termination;
    std::shared_ptr<material> top;
    std::shared_ptr<Medium> medium;
    std::shared_ptr<material> bottom;
};

#endif
