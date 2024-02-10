#ifndef HITTABLE_H
#define HITTABLE_H

#include "general.h"
#include "aabb.h"
#include "interval.h"

class material;
class timeline;

struct hit_record {
    point3 p;
    vec3 normal;
    shared_ptr<material> mat_ptr;
    double t;
    bool front_face;

    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
    public:
        hittable() {}
        hittable(shared_ptr<timeline> t) : t_line(t) {}
        virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
        virtual aabb bounding_box() const = 0;
    public:
        shared_ptr<timeline> t_line;
};

#endif