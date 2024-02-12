
#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

#include "vec3.h"
#include "aabb.h"

class sphere : public hittable {
    public:
        sphere(point3 cen, double r, shared_ptr<material> m, shared_ptr<timeline> t) 
            : center1(cen), center2(t->motion.back().position), radius(r), mat_ptr(m), hittable(t) {
                auto rvec = vec3(radius, radius, radius);
                aabb bbox1 = aabb(center1 - rvec, center1 + rvec);
                aabb bbox2 = aabb(center2 - rvec, center2 + rvec);
                bbox = aabb(bbox1, bbox2);
            }

        virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
        virtual aabb bounding_box() const override { return bbox; }
    public:
        point3 center1;
        point3 center2;
        double radius;
        shared_ptr<material> mat_ptr;
        aabb bbox;
 /* private: */
        static void get_sphere_uv(const point3& p, double& u, double& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + pi;

        u = phi / (2*pi);
        v = theta / pi;
    }
};


bool sphere::hit(const ray& r, interval ray_t, hit_record& rec) const {
    vec3 new_center = t_line->interpolate_position(r.time());
    //vec3 new_center = center;
    //std::cerr << "\n" << center;
    //std::cerr << t_line->interpolate_position(r.time());

    vec3 oc = r.origin() - new_center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    auto root = (-half_b - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (-half_b + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    } // both roots must lie within the range

    rec.t = root; // time of the ray?
    rec.p = r.at(rec.t); // rec.t is the time value, passed to find the point of intersection
    rec.normal = (rec.p - new_center) / radius; //normal vector from the sphere
    vec3 outward_normal = (rec.p - new_center) / radius;
    rec.set_face_normal(r, outward_normal);
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;

    return true;
}

#endif

