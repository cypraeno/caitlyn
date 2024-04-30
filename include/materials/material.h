#ifndef MATERIAL_H
#define MATERIAL_H

#include "general.h"
#include "hitinfo.h"
#include "texture.h"
#include "onb.h"

#include <complex>
#include "microfacet.h"

class hit_record;

// CONSTANTS
const float SPECULAR_ROUGHNESS_SAMPLING_CUTOFF = 0.1;

enum BSDF_TYPE {
    DIFFUSE,
    GLOSSY,
    SPECULAR,
    TRANSMISSION
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
        virtual color emitted(double u, double v, const point3& p) const {
            return color(0,0,0);
        }

        virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const {
            return true;
        }
        virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
            return color(0,0,0);
        }
	    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
            return 1.0;
        };

        virtual BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const {
            BSDFSample sample_data;
            // Sample the microfacet distribution to get the scatter direction.
            color attenuation; // placeholder until it gets removed from the scatter function header
            sample_data.scatter = scatter(r_in, rec, attenuation, scattered);
            sample_data.scatter_direction = scattered.direction().unit_vector();

            // Sample the BRDF for the value
            sample_data.bsdf_value = generate(r_in, scattered, rec);

            // Find the PDF for the MDF
            sample_data.pdf_value = pdf(r_in, scattered, rec);
            return sample_data;
        }
};

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

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const override {
        onb uvw;
        uvw.build_from_w(rec.normal);
        auto scatter_direction = uvw.local(random_cosine_direction());
        scattered = ray(rec.pos, scatter_direction, r_in.time());
        
        return true;
    }

    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
        vec3 w_i = scattered.direction().unit_vector();
        vec3 w_o = -(r_in.direction().unit_vector());

        // Calculate azimuthal angles.
        vec3 projected_i = (w_i - (dot(w_i, rec.normal) * rec.normal)).unit_vector();
        vec3 projected_o = (w_o - (dot(w_o, rec.normal) * rec.normal)).unit_vector();
        float cos_azimuth = dot(projected_i, projected_o);


        float theta_i = acos(dot(w_i, rec.normal));
        float theta_o = acos(dot(w_o, rec.normal));
        
        float sigma2 = roughness * roughness;
        float A = 1 - (sigma2 / (2 * (sigma2 + 0.33)));

        float B = (0.45 * sigma2) / (sigma2 + 0.09);

        float alpha = fmax(theta_i, theta_o);
        float beta = fmin(theta_i, theta_o);

        color diffuse_term = albedo / pi;


        return diffuse_term * (A + B * (fmax(0, cos_azimuth) * sin(alpha) * tan(beta)));
    }

    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
        auto cos_theta = dot(rec.normal, scattered.direction().unit_vector());
        return fmax(0.0, cos_theta / pi);
    }

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

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const override {
        vec3 microfacet_normal = MDF->sample(rec.normal);
        rec.microfacet_normal = microfacet_normal;
        vec3 scatter_direction = reflect(r_in.direction().unit_vector(), microfacet_normal);
        scattered = ray(rec.pos, scatter_direction, r_in.time());

        return (dot(scattered.direction(), rec.normal) > 0);
    }

    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
        vec3 L = scattered.direction().unit_vector();
        vec3 N = rec.normal;
        vec3 V = -(r_in.direction().unit_vector());
        vec3 H = (V + L).unit_vector();

        float NoV = clamp(dot(N, V), 0.0, 1.0);
        float NoL = clamp(dot(N, L), 0.0, 1.0);
        float NoH = clamp(dot(N, H), 0.0, 1.0);
        float VoH = clamp(dot(V, H), 0.0, 1.0);

        vec3 f0 = albedo; vec3 F;
        if (complex) { F = FrComplex(fabs(dot(V,rec.microfacet_normal)), absorption_coefficient, eta); }
        else { F = fresnelSchlick(VoH, f0); }

        float D = MDF->D(NoH);
        float G = MDF->G(NoV, NoL);

        vec3 spec = (F * D * G) / (4.0 * fmax(NoV, 0.001) * fmax(NoL, 0.001));

        return spec;
    }

    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const override {
        vec3 V = -r_in.direction().unit_vector();
        vec3 L = scattered.direction().unit_vector();
        vec3 H = (V + L).unit_vector();
        vec3 N = rec.normal;

        float NoH = clamp(dot(N, H), 0.0, 1.0);
        float VoH = clamp(dot(V, H), 0.0, 1.0);

        float D = MDF->D(NoH);
        // Convert D(N·H) to pdf based on the microfacet normal distribution.
        // The Jacobian of the half-vector reflection transformation is |4 * (V·H)|.
        // This accounts for the change in area density when mapping from H to L.
        float jacobian = 4.0 * abs(dot(V, H));
        if (jacobian < 0.0001) return 0;

        return D / jacobian;
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
        if (MDF->roughness < SPECULAR_ROUGHNESS_SAMPLING_CUTOFF) { sample_data.type = BSDF_TYPE::SPECULAR; } // 0.05 was picked arbitrarily, should experiment
        else { sample_data.type = BSDF_TYPE::GLOSSY; }
        return sample_data;
    }

    private:
    bool complex;
    color albedo;
    color absorption_coefficient;
    color eta;
    std::shared_ptr<Microfacet> MDF;

    // F, G, D functions
    vec3 fresnelSchlick(float cosTheta, vec3 F0) const {
        return F0 + (color(1.0, 1.0, 1.0) - F0) * pow(1.0 - cosTheta, 5.0);
    }

    float FrComplex(float cosTheta_i, std::complex<float> eta) const {
        using Complex = std::complex<float>;
        cosTheta_i = clamp(cosTheta_i, 0, 1);
        float sin2Theta_i = 1 - (cosTheta_i * cosTheta_i);
        Complex sin2Theta_t = sin2Theta_i / (eta * eta);
        Complex val(1, -2);
        Complex cosTheta_t = std::sqrt(val - sin2Theta_t);
        
        Complex r_parl = (eta * cosTheta_i - cosTheta_t) /
                     (eta * cosTheta_i + cosTheta_t);
        Complex r_perp = (cosTheta_i - eta * cosTheta_t) /
                        (cosTheta_i + eta * cosTheta_t);
        return (std::norm(r_parl) + std::norm(r_perp)) / 2;
    }

    vec3 FrComplex(float cosTheta_v, vec3 k, vec3 eta) const {
        float x = FrComplex(cosTheta_v, std::complex<float>(eta.x(), k.x()));
        float y = FrComplex(cosTheta_v, std::complex<float>(eta.y(), k.y()));
        float z = FrComplex(cosTheta_v, std::complex<float>(eta.z(), k.z()));
        return vec3(x,y,z);
    }

};

class CookTorranceDielectric : public material {

    public:
    CookTorranceDielectric(color albedo, float eta, float roughness) : albedo{albedo}, eta{(eta == 0.0f) ? 0.0f : (float)fmax(eta, 1.0001f)}, MDF{std::make_shared<GGX>(roughness)} {}

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const {
        vec3 wo = -r_in.direction().unit_vector();
        vec3 N = rec.normal;
        vec3 wm = MDF->sample(N); // outward microfacet normal.
        rec.microfacet_normal = wm;

        float cosTheta_i = dot(wo, wm);
        float R = FrDielectric(cosTheta_i);
        float T = 1 - R;

        float u = random_double();
        double refraction_ratio = rec.front_face ? (1.0/eta) : eta;
        double sinTheta_i = sqrt(1.0 - cosTheta_i*cosTheta_i);

        if (u < (R / (R + T)) || refraction_ratio * sinTheta_i > 1.0) { // reflectance
            vec3 wi = reflect(-wo, wm);
            scattered = ray(rec.pos, wi, r_in.time());
        } else {
            vec3 wi = refract(-wo, wm, refraction_ratio);
            scattered = ray(rec.pos, wi, r_in.time());
        }
        return true;
    }
    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const { // assumes wm has been defined in rec
        vec3 wo = -r_in.direction().unit_vector();
        vec3 N = rec.normal;
        vec3 wi = scattered.direction();
        vec3 wm = rec.microfacet_normal;
        float cosTheta_i = dot(wo, wm);
        float R = FrDielectric(cosTheta_i);
        float T = 1 - R;
        if (cosTheta_i > 0) { // reflectance
            return f_r(r_in, rec, scattered, R);
        } else { // refractance
            return f_t(r_in, rec, scattered, T);
        }
    }
    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const { // assumes wm has been defined in rec
        vec3 wo = -r_in.direction().unit_vector();
        vec3 N = rec.normal;
        vec3 wi = scattered.direction();
        vec3 wm = rec.microfacet_normal;
        float cosTheta_i = dot(wo, wm);
        float R = FrDielectric(cosTheta_i);
        float T = 1 - R;
        if (cosTheta_i > 0) { // reflectance
            return pdf_r(r_in, rec, scattered, R);
        } else { // refractance
            return pdf_t(r_in, rec, scattered, T);
        }
    };
    
    BSDFSample sample(const ray& r_in, HitInfo& rec, ray& scattered) const override {
        // Vectors wo and wi are the outgoing and incident directions respectively.
        vec3 wo = -r_in.direction().unit_vector();

        BSDFSample sample_data;
        vec3 N = rec.normal;
        vec3 wm = MDF->sample(N); // outward microfacet normal.
        rec.microfacet_normal = wm;

        float cosTheta_i = dot(wo, wm);
        float R = FrDielectric(cosTheta_i);
        float T = 1 - R;

        float u = random_double();
        double refraction_ratio = rec.front_face ? (1.0/eta) : eta;
        double sinTheta_i = sqrt(1.0 - cosTheta_i*cosTheta_i);

        if (u < (R / (R + T)) || refraction_ratio * sinTheta_i > 1.0) { // reflectance
            vec3 wi = reflect(-wo, wm);
            scattered = ray(rec.pos, wi, r_in.time());
            sample_data.scatter_direction = wi;
            sample_data.scatter = (dot(wi, N) > 0);

            sample_data.bsdf_value = f_r(r_in, rec, scattered, R);
            sample_data.pdf_value = pdf_r(r_in, rec, scattered, R);
            if (MDF->roughness < SPECULAR_ROUGHNESS_SAMPLING_CUTOFF) { sample_data.type = BSDF_TYPE::SPECULAR; } // 0.05 was picked arbitrarily, should experiment
            else { sample_data.type = BSDF_TYPE::GLOSSY; }
        } else { // transmission
            vec3 wi = refract(-wo, wm, refraction_ratio);
            scattered = ray(rec.pos, wi, r_in.time());
            sample_data.scatter_direction = wi;
            sample_data.scatter = (dot(wi, N) < 0);

            sample_data.bsdf_value = f_t(r_in, rec, scattered, T);
            sample_data.pdf_value = pdf_t(r_in, rec, scattered, T);
            sample_data.type = BSDF_TYPE::TRANSMISSION;
        }

        return sample_data;
    }

    //private:
    color albedo;
    float eta;
    std::shared_ptr<Microfacet> MDF;

    float FrDielectric(float cosTheta_i) const {
        float temp_eta = eta;
        cosTheta_i = clamp(cosTheta_i, -1.0, 1.0);
        if (cosTheta_i < 0) {
            temp_eta = 1 / eta;
            cosTheta_i = -cosTheta_i;
        }

        float sin2Theta_i = 1 - (cosTheta_i * cosTheta_i);
        float sin2Theta_t = sin2Theta_i / (temp_eta * temp_eta);
        if (sin2Theta_t >= 1.0) {
            return 1.0;
        }
        float cosTheta_t = sqrt(1 - sin2Theta_t);
        float r_parallel = (temp_eta * cosTheta_i - cosTheta_t) / (temp_eta * cosTheta_i + cosTheta_t);
        float r_perp = (cosTheta_i - (temp_eta * cosTheta_t)) / (cosTheta_i + (eta * cosTheta_t));
        return ((r_parallel * r_parallel) + (r_perp * r_perp)) / 2;
    }
    private:
    color f_r(const ray& r_in, const HitInfo& rec, const ray& scattered, float R) const {
        vec3 V = -r_in.direction().unit_vector();
        vec3 L = scattered.direction().unit_vector();
        vec3 H = (V + L).unit_vector();
        vec3 N = rec.normal;

        float NoH = clamp(dot(N, H), 0.0, 1.0);
        float NoV = clamp(dot(N, V), 0.0, 1.0);
        float NoL = clamp(dot(N, L), 0.0, 1.0);

        float D = MDF->D(NoH);
        float G = MDF->G(NoV, NoL);
        
        color R_col = color(R, R, R) * albedo;

        color num = D * G * R_col;
        float denom = (4.0 * fmax(NoV, 0.001) * fmax(NoL, 0.001));

        return num / denom;
    }

    float pdf_r(const ray& r_in, const HitInfo& rec, const ray& scattered, float R) const {
        vec3 V = -r_in.direction().unit_vector();
        vec3 L = scattered.direction().unit_vector();
        vec3 H = (V + L).unit_vector();
        vec3 N = rec.normal;

        float NoH = clamp(dot(N, H), 0.0, 1.0);
        float VoH = clamp(dot(V, H), 0.0, 1.0);

        float D = MDF->D(NoH);
        // Convert D(N·H) to pdf based on the microfacet normal distribution.
        // The Jacobian of the half-vector reflection transformation is |4 * (V·H)|.
        // This accounts for the change in area density when mapping from H to L.
        float jacobian = 4.0 * abs(dot(V, H));
        if (jacobian < 0.0001) return 0;

        return (D * R) / jacobian;
    }

    float pdf_t(const ray& r_in, const HitInfo& rec, const ray& scattered, float T) const {
        double etap = rec.front_face ? (1.0/eta) : eta;

        vec3 wo = -r_in.direction().unit_vector();
        vec3 wi = scattered.direction().unit_vector();
        vec3 wn = rec.normal;
        vec3 wm = rec.microfacet_normal;
        vec3 h = (wo + wi).unit_vector();

        float denom = (dot(wi, wm) + dot(wo, wm) / etap) * (dot(wi, wm) + dot(wo, wm) / etap);
        float dwm_dwi = fabs(dot(wi, wm)) / denom;
        float NoM = dot(wm, wn);
        float D = MDF->D(NoM);
        return D * dwm_dwi * T;
    }

    color f_t(const ray& r_in, const HitInfo& rec, const ray& scattered, float T) const {
        double etap = rec.front_face ? (1.0/eta) : eta;

        vec3 wo = -r_in.direction().unit_vector();
        vec3 wi = scattered.direction().unit_vector();
        vec3 wn = rec.normal;
        vec3 wm = rec.microfacet_normal;
        vec3 h = (wo + wi).unit_vector();

        float NoM = dot(wm, wn);
        float NoO = dot(wn, wo);
        float NoI = dot(wn, wi);
        float D = MDF->D(NoM);
        float G = MDF->G(fabs(NoO), fabs(NoI));
        color T_col = color(T, T, T) * albedo;
        color num = D * G * T_col;

        float IoM = dot(wi, wm);
        float OoM = dot(wo, wm);
        float denom = (IoM + OoM / etap) * (IoM + OoM / etap);
        float dotabs = fabs(IoM * OoM / (dot(wi, wn) * dot(wo, wn) * denom)); // 1: e+14, 2: inf
        return num * dotabs;
    } 
};

class isotropic : public material {
    isotropic(const color& albedo) : tex(make_shared<solid_color>(albedo)) {}
    isotropic(shared_ptr<texture> tex) : tex(tex) {}

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const {
        scattered = ray(rec.p, random_unit_vector(), r_in.time());
        return true;
    }
    virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
        return 1 / (4 * pi);
    }
    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
        return 1 / (4 * pi);
    };
}

#endif
