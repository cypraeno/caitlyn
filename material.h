#ifndef MATERIAL_H
#define MATERIAL_H

#include "general.h"
#include "hitinfo.h"
#include "texture.h"
#include "onb.h"

#include <complex>

class hit_record;

class material {

    public:
        virtual color emitted(double u, double v, const point3& p) const {
            return color(0,0,0);
        }

        virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const = 0;
        virtual color generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
            return color(0,0,0);
        }
	    virtual double pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
            return 1.0;
        };
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
*/
class CookTorrance : public material {

    public:
    CookTorrance(color albedo, float roughness)
        : complex{false}, albedo{albedo}, roughness{roughness} {}

    CookTorrance(float roughness, color absorption, color refraction)
        : complex(true), roughness{roughness}, absorption_coefficient{absorption}, eta{refraction} {}

    virtual bool scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const override {
        vec3 microfacet_normal = random_GGX_microfacet(rec.normal);
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

        float D = D_GGX(NoH, roughness);
        float G = G_Smith(NoV, NoL, roughness);

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

        float D = D_GGX(NoH, roughness);
        // Convert D(N·H) to pdf based on the microfacet normal distribution.
        // The Jacobian of the half-vector reflection transformation is |4 * (V·H)|.
        // This accounts for the change in area density when mapping from H to L.
        float jacobian = 4.0 * abs(dot(V, H));
        if (jacobian < 0.0001) return 0;

        return D / jacobian;
    }

    private:
    bool complex;
    color albedo;
    float roughness; // 0-1
    color absorption_coefficient; // RGB value of how much to not absorb. The higher the color channel, the less that one is absorbed.
    color eta; // RGB value of how much to refract (i.e NOT reflect). The higher the color channel, the less it shows.

    // F, G, D functions
    vec3 fresnelSchlick(float cosTheta, vec3 F0) const {
        return F0 + (color(1.0, 1.0, 1.0) - F0) * pow(1.0 - cosTheta, 5.0);
    }

    float D_GGX(float NoH, float roughness) const {
        float alpha = roughness * roughness;
        float alpha2 = alpha * alpha;
        float NoH2 = NoH * NoH;
        float b = (NoH2 * (alpha2 - 1.0) + 1.0);
        return (alpha2 / pi) / (b * b);
    }

    float G1_GGX_Schlick(float NoV, float roughness) const {
        float alpha = roughness * roughness;
        float k = alpha / 2.0;
        return fmax(NoV, 0.001) / (NoV * (1.0 - k) + k);
    }

    float G_Smith(float NoV, float NoL, float roughness) const {
        return G1_GGX_Schlick(NoL, roughness) * G1_GGX_Schlick(NoV, roughness);
    }

    vec3 random_GGX_microfacet(vec3 N) const {
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


#endif
