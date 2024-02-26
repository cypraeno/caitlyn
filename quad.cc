#include "quad.h"

quad::quad(const point3& _Q, const vec3& _u, const vec3& _v, shared_ptr<material> _mat) :
    Q(_Q), u(_u), v(_v), mat(_mat) {
        auto n = cross(u, v);
        this->normal = n.unit_vector();
        this->D = dot(normal, Q);
        this->w = n / dot(n, n);

        this->set_bounding_box();
    }

virtual void quad::set_bounding_box() {
    this->bbox = aabb(Q, Q + u + v).pad();
}

double quad::get_D() const { return this->D; }
point3 quad::get_Q() const { return this->Q; }
vec3 quad::get_u() const { return this->u; }
vec3 quad::get_v() const { return this->v; }
vec3 quad::get_w() const { return this->w; }
vec3 quad::get_normal() const { return this->normal; }
shared_ptr<material> get_mat() const { return this->mat; }
aabb quad::bounding_box() const override { return this->bbox; }

bool quad::hit(const ray& r, interval ray_t, hit_record& rec) const override {
    
    // return false (no hit) if ray parallel to plane (dot product ~0)
    float denom = dot(this->normal, r.direction());
    if (fabs(denom) < 1e-8) return false;

    // return false if hit point (t) outside interval (ray_t)
    double t = (this->D - dot(normal, r.origin())) / denom;
    if (!ray_t.contains(t)) return false;

    point3 intersection = r.at(t);

    rec.t = t;
    rec.p = intersection;
    rec.mat = this->mat;
    rec.set_face_normal(r, this->normal);

    return true;
}
