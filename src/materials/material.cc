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

bool CookTorrance::scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const {
    vec3 microfacet_normal = MDF->sample(rec.normal);
    rec.microfacet_normal = microfacet_normal;
    vec3 scatter_direction = reflect(r_in.direction().unit_vector(), microfacet_normal);
    scattered = ray(rec.pos, scatter_direction, r_in.time());

    return (dot(scattered.direction(), rec.normal) > 0);
}

color CookTorrance::generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
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

double CookTorrance::pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
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

BSDFSample CookTorrance::sample(const ray& r_in, HitInfo& rec, ray& scattered) const {
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

vec3 CookTorrance::fresnelSchlick(float cosTheta, vec3 F0) const {
    return F0 + (color(1.0, 1.0, 1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float CookTorrance::FrComplex(float cosTheta_i, std::complex<float> eta) const {
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

vec3 CookTorrance::FrComplex(float cosTheta_v, vec3 k, vec3 eta) const {
    float x = FrComplex(cosTheta_v, std::complex<float>(eta.x(), k.x()));
    float y = FrComplex(cosTheta_v, std::complex<float>(eta.y(), k.y()));
    float z = FrComplex(cosTheta_v, std::complex<float>(eta.z(), k.z()));
    return vec3(x,y,z);
}

// ===================== COOK DIELECTRIC ===============================

bool CookTorranceDielectric::scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const {
    vec3 wo = -r_in.direction().unit_vector();
    vec3 N = rec.normal;
    vec3 wm = MDF->sample(N); // outward microfacet normal.
    rec.microfacet_normal = wm;

    float cosTheta_i = dot(wo, wm);
    float R;
    if (complexFresnel == 0) { R = FrDielectric(cosTheta_i); }
    else { R = fresnelSchlick(cosTheta_i, complexFresnel); }
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
color CookTorranceDielectric::generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const { // assumes wm has been defined in rec
    vec3 wo = -r_in.direction().unit_vector();
    vec3 N = rec.normal;
    vec3 wi = scattered.direction();
    vec3 wm = rec.microfacet_normal;
    float cosTheta_i = dot(wo, wm);
    float R;
    if (complexFresnel == 0) { R = FrDielectric(cosTheta_i); }
    else { R = fresnelSchlick(cosTheta_i, complexFresnel); }
    float T = 1 - R;
    if (cosTheta_i > 0) { // reflectance
        return f_r(r_in, rec, scattered, R);
    } else { // refractance
        return f_t(r_in, rec, scattered, T);
    }
}
double CookTorranceDielectric::pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const { // assumes wm has been defined in rec
    vec3 wo = -r_in.direction().unit_vector();
    vec3 N = rec.normal;
    vec3 wi = scattered.direction();
    vec3 wm = rec.microfacet_normal;
    float cosTheta_i = dot(wo, wm);
    float R;
    if (complexFresnel == 0) { R = FrDielectric(cosTheta_i); }
    else { R = fresnelSchlick(cosTheta_i, complexFresnel); }
    float T = 1 - R;
    if (cosTheta_i > 0) { // reflectance
        return pdf_r(r_in, rec, scattered, R);
    } else { // refractance
        return pdf_t(r_in, rec, scattered, T);
    }
};

BSDFSample CookTorranceDielectric::sample(const ray& r_in, HitInfo& rec, ray& scattered) const {
    // Vectors wo and wi are the outgoing and incident directions respectively.
    vec3 wo = -r_in.direction().unit_vector();

    BSDFSample sample_data;
    vec3 N = rec.normal;
    vec3 wm = MDF->sample(N); // outward microfacet normal.
    rec.microfacet_normal = wm;

    float cosTheta_i = dot(wo, wm);
    float R;
    if (complexFresnel == 0) { R = FrDielectric(cosTheta_i); }
    else { R = fresnelSchlick(cosTheta_i, complexFresnel); }
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

float CookTorranceDielectric::FrDielectric(float cosTheta_i) const {
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

float CookTorranceDielectric::fresnelSchlick(float cosTheta, int exponent) const {
    float F0 = pow(((1 - eta) / (1 + eta)), 2);
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, exponent);
}

color CookTorranceDielectric::f_r(const ray& r_in, const HitInfo& rec, const ray& scattered, float R) const {
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

float CookTorranceDielectric::pdf_r(const ray& r_in, const HitInfo& rec, const ray& scattered, float R) const {
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

float CookTorranceDielectric::pdf_t(const ray& r_in, const HitInfo& rec, const ray& scattered, float T) const {
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

color CookTorranceDielectric::f_t(const ray& r_in, const HitInfo& rec, const ray& scattered, float T) const {
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
bool MixtureBSDF::scatter(const ray& r_in, HitInfo& rec, color& attenuation, ray& scattered) const {
    rec.rand = random_double();
    int mat_ix = chooseSampleMaterial(rec.rand);
    if (mat_ix >= 0) { // weights is valid
        return mats[mat_ix]->scatter(r_in, rec, attenuation, scattered);
    } else {
        return false;
    }
}

// assumes that scatter has already been called or sample has already been called, and thus rand is already generated.
color MixtureBSDF::generate(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
    int mat_ix = chooseSampleMaterial(rec.rand);
    if (mat_ix >= 0) { return mats[mat_ix]->generate(r_in, scattered, rec);
    } else { return color(1,1,1); }
}

double MixtureBSDF::pdf(const ray& r_in, const ray& scattered, const HitInfo& rec) const {
    int mat_ix = chooseSampleMaterial(rec.rand);
    if (mat_ix >= 0) { return mats[mat_ix]->pdf(r_in, scattered, rec);
    } else { return 1.0; }
}

// NOTE: ASSUMES WEIGHTS IS AT LEAST OF SIZE 1 OTHERWISE BEHAVIOUR IS UNDEFINED
BSDFSample MixtureBSDF::sample(const ray& r_in, HitInfo& rec, ray& scattered) const {
    rec.rand = random_double();
    int mat_ix = chooseSampleMaterial(rec.rand);
    if (mat_ix >= 0) { return mats[mat_ix]->sample(r_in, rec, scattered);
    } else {
        BSDFSample sample_data;
        return sample_data;
    }
}
int MixtureBSDF::chooseSampleMaterial(float rand) const {
    float cumulative_weight = 0.0f;
    for (size_t i = 0; i < weights.size(); i++) {
        cumulative_weight += weights[i];
        if (rand < cumulative_weight) {
            return i;
        }
    }
    return (int)weights.size() - 1;
}
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