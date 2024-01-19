#ifndef SPHERE_H
#define SPHERE_H

#include "general.h"
#include "hittable.h"


class sphere : public hittable {
    public:
        sphere() {}
        sphere(point3 cen, double r, shared_ptr<material> m, shared_ptr<timeline> t) 
            : centre(cen), radius(r), mat_ptr(m), hittable(t) {};

        virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
    public:
        point3 centre;
        double radius;
        shared_ptr<material> mat_ptr;
};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    vec3 new_centre = t_line->interpolate_position(r.time());
    //vec3 new_centre = centre;
    //std::cerr << "\n" << centre;
    //std::cerr << t_line->interpolate_position(r.time());

    vec3 oc = r.origin() - new_centre;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    } // both roots must lie within the range

    rec.t = root; // time of the ray?
    rec.p = r.at(rec.t); // rec.t is the time value, passed to find the point of intersection
    rec.normal = (rec.p - new_centre) / radius; //normal vector from the sphere
    vec3 outward_normal = (rec.p - new_centre) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;

}

#endif