#ifndef MATERIAL_H
#define MATERIAL_H

#include "general.h"
#include "hitinfo.h"
#include "texture.h"

class hit_record;

class material {

    public:
        virtual color emitted(double u, double v, const point3& p) const {
            return color(0,0,0);
        }

        virtual bool scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {

    public:
        lambertian(const color& a) : albedo(make_shared<solid_color>(a)) {}
        lambertian(shared_ptr<texture> a) : albedo(a) {}

        virtual bool scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const override {

            auto scatter_direction = rec.normal + random_unit_vector();

            if (scatter_direction.near_zero()) {
                scatter_direction = rec.normal;
            }
            scattered = ray(rec.pos, scatter_direction, r_in.time());
            attenuation = albedo->value(rec.u, rec.v, rec.pos);
            
            return true;
        }

    private:
    shared_ptr<texture> albedo;
};

class hemispheric : public material {

    public:

        hemispheric(const color& a) : albedo(a) {}

        virtual bool scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const override {
            auto scatter_direction = random_in_hemisphere(rec.normal);

            if (scatter_direction.near_zero()) {
                scatter_direction = rec.normal;
            }
            scattered = ray(rec.pos,scatter_direction, r_in.time());
            attenuation = albedo;

            return true;
        }

    public:

        color albedo;
};

class metal : public material {

    public:

        metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        virtual bool scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const override {
            vec3 reflected = reflect(r_in.direction().unit_vector(), rec.normal);
            scattered = ray(rec.pos, reflected + fuzz*random_in_unit_sphere(), r_in.time());
            attenuation = albedo;

            return (dot(scattered.direction(), rec.normal) > 0);
        }

    public:

        color albedo;
        double fuzz;
};

class dielectric : public material {

    public:

        dielectric(double index_of_refraction) : ir(index_of_refraction) {}

        virtual bool scatter(const ray& r_in, const HitInfo& rec, color& attenuation, ray& scattered) const override {
            attenuation = color(1.0, 1.0, 1.0);
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

#endif
