#include "material.h"

#include "medium.h"

color material::emitted(double u, double v, const point3& p) const { return color(0,0,0); }
bool material::scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const { return true; }
color material::generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const { return color(0,0,0); }
double material::pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const { return 1.0; };

// Default behaviour for sample unless overriden by material
BSDFSample material::sample(const ray& r_in, HitInfo& rec, ray& scattered) const {
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


// ===================== OREN NAYAR ===============================
bool OrenNayar::scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const {
    onb uvw;
    uvw.build_from_w(rec.normal);
    auto scatter_direction = uvw.local(random_cosine_direction());
    scattered = ray(rec.pos, scatter_direction, r_in.time());
    
    return true;
}

color OrenNayar::generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
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

double OrenNayar::pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
    auto cos_theta = dot(rec.normal, scattered.direction().unit_vector());
    return fmax(0.0, cos_theta / pi);
}

// ===================== COOK TORRANCE ===============================
// ===================== COOK DIELECTRIC ===============================

// ===================== ISOTROPIC ===============================
bool isotropic::scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const {
    scattered = ray(rec.pos, random_unit_vector(), r_in.time());
    return true;
}
color isotropic::generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
    return albedo / (4 * pi);
}
double isotropic::pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
    return 1 / (4 * pi);
};
// ===================== PIXEL LAMBERTIAN ===============================
bool pixel_lambertian::scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const{
    float t = random_double();
    color4 val = albedo->value(rec.u, rec.v);
    if (t > val.A) { // transparent
        rec.transparent = true;
        scattered = ray(rec.pos, r_in.direction(), 0.0);
    } else {
        onb uvw;
        uvw.build_from_w(rec.normal);
        auto scatter_direction = uvw.local(random_cosine_direction());
        scattered = ray(rec.pos, scatter_direction, r_in.time());
    }
    
    
    return true;
}

color pixel_lambertian::generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
    if (!rec.transparent) {
        return albedo->value(rec.u, rec.v).RGB / pi;
    } else {
        return color(1.0, 1.0, 1.0) / pi;
    }
}

double pixel_lambertian::pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
    auto cos_theta = dot(rec.normal, scattered.direction().unit_vector());
    if (!rec.transparent) { return fmax(0.0, cos_theta / pi); }
    else { return fabs(cos_theta) / pi; }
}

BSDFSample pixel_lambertian::sample(const ray& r_in, HitInfo& rec, ray& scattered) const {
    BSDFSample sample_data;
    // Sample the microfacet distribution to get the scatter direction.
    color attenuation; // placeholder until it gets removed from the scatter function header
    sample_data.scatter = scatter(r_in, rec, attenuation, scattered);
    sample_data.scatter_direction = scattered.direction().unit_vector();

    // Sample the BRDF for the value
    sample_data.bsdf_value = generate(r_in, scattered, rec);

    // Find the PDF for the MDF
    sample_data.pdf_value = pdf(r_in, scattered, rec);
    if (rec.transparent) { sample_data.type = BSDF_TYPE::TRANSPARENT; }
    return sample_data;
}

// ===================== MIXTURE ===============================
// ===================== LAYERED ===============================
BSDFSample LayeredBSDF::sample(const ray& r_in, HitInfo& rec, ray& scattered) const {
    // Return in case calculating a full simulation becomes impossible or irrelevant
    BSDFSample absorbed; absorbed.scatter = false;
    
    ray r = r_in;
    HitInfo rec_manip = rec;
    vec3 outward_normal = rec.front_face ? rec.normal : -rec.normal;

    bool on_top = rec_manip.front_face;

    std::shared_ptr<material> current = on_top ? top : bottom;
    BSDFSample bs = current->sample(r, rec_manip, scattered);

    color f = bs.bsdf_value;
    float pdf = bs.pdf_value;

    float distance_in_from_top = rec.front_face ? 0.0 : 1.0; // amount of distance ray has traveled into between the layers
    bool prev_medium = false; // previous hit was a medium

    if (!bs.scatter) { return absorbed; }
    if (bs.type != BSDF_TYPE::TRANSMISSION && bs.type != BSDF_TYPE::TRANSPARENT) {
        bs.type = (dot(bs.scatter_direction, rec.normal) < 0) ? BSDF_TYPE::TRANSMISSION : BSDF_TYPE::SPECULAR;
        return bs;
    }
    for (int depth = 0; depth < termination; depth++) {
        
        // Follow random walk through layers to sample layered BSDF
        on_top = !on_top;
        current = on_top ? top : bottom;

        // Possibly terminate layered BSDF sampling with Russian Roulette
        float rrBeta = fmax(fmax(f.x(), f.y()), f.z()) / bs.pdf_value;
        if (depth > 3 && rrBeta < 0.25) { // rrBeta < 0.50 reduces by more, but probably reduces accuracy
            float q = fmax(0, 1-rrBeta);
            if (random_double() < q) {
                if (on_top == rec.front_face) {
                    bs.type = (dot(bs.scatter_direction, rec.normal) < 0) ? BSDF_TYPE::TRANSMISSION : BSDF_TYPE::SPECULAR;
                    return bs;
                }
            }
            pdf *= 1 - q;
        }

        if (!prev_medium) { f = f * fabs(dot(rec_manip.normal, (bs.scatter_direction))); }
        prev_medium = false;
        r = ray(rec_manip.pos - bs.scatter_direction, bs.scatter_direction, r.time());

        if (medium) {
            float distance = medium->particleDistance();
            float distance_to_barrier;
            if (on_top) { distance_to_barrier = distance_in_from_top; }
            else { distance_to_barrier = thickness - distance_in_from_top; }
            if (distance < distance_to_barrier) { // collide with particle
                distance_in_from_top += (on_top * -distance) + (!on_top * distance);
                bs = medium->phase->sample(r, rec_manip, scattered);
                prev_medium = true;

                f = f * bs.bsdf_value;
                pdf = pdf * bs.pdf_value;
                bs.bsdf_value = f;
                bs.pdf_value = pdf;

                if (dot(bs.scatter_direction, outward_normal) > 0) {
                    on_top = false; // because it will flip at the start of the next loop
                    rec_manip.front_face = false;
                    rec_manip.normal = -outward_normal;
                } else {
                    on_top = true; // because it will flip at the start of the next loop
                    rec_manip.front_face = true;
                    rec_manip.normal = outward_normal;
                }

                continue;
            } else {
                if (on_top) { distance_in_from_top = 0.0;} else { distance_in_from_top = 1.0; }
            }
        }

        bs = current->sample(r, rec_manip, scattered);
        f = f * bs.bsdf_value;
        pdf = pdf * bs.pdf_value;
        bs.bsdf_value = f;
        bs.pdf_value = pdf;

        if (bs.type == BSDF_TYPE::TRANSMISSION) {
            bs.type = (dot(bs.scatter_direction, rec.normal) < 0) ? BSDF_TYPE::TRANSMISSION : BSDF_TYPE::SPECULAR;
            return bs;
        }

        // Flip since coming from the bottom!
        rec_manip.front_face = !rec_manip.front_face;
        rec_manip.normal = -rec_manip.normal;
    }
    bs.type = (dot(bs.scatter_direction, rec.normal) < 0) ? BSDF_TYPE::TRANSMISSION : BSDF_TYPE::SPECULAR;
    return bs;
}