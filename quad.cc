#include "quad.h"

quad::quad(const point3& _Q, const vec3& _u, const vec3& _v, shared_ptr<material> _mat) :
    Q(_Q), u(_u), v(_v), mat(_mat) {
        auto n = cross(u, v);
        this->normal = n.unit_vector();
        this->D = dot(normal, Q);
        this->w = n / dot(n, n);

        this->set_bounding_box();
    }

void quad::set_bounding_box() {
    this->bbox = aabb(Q, Q + u + v).pad();
}

bool quad::is_interior(double a, double b, hit_record& rec) const {
    
    if ((a < 0) || (1 < a) || (b < 0) || (1 < b)) return false;

    rec.u = a;
    rec.v = b;
    return true;
}

bool quad::hit(const ray& r, interval ray_t, hit_record& rec) const {
    
    // return false (no hit) if ray parallel to plane (dot product ~0)
    float denom = dot(this->normal, r.direction());
    if (fabs(denom) < 1e-8) return false;

    // return false if hit point (t) outside interval (ray_t)
    double t = (this->D - dot(normal, r.origin())) / denom;
    if (!ray_t.contains(t)) return false;

    // return false if hit point lies outside planar shape
    point3 intersection = r.at(t);
    vec3 planar_hitpoint = intersection - this->Q;
    float alpha = dot(this->w, cross(planar_hitpoint, this->v));
    float beta = dot(this->w, cross(this->u, planar_hitpoint));

    if (!this->is_interior(alpha, beta, rec)) return false;

    // ray hits 2D shape: set hit record and return true
    rec.t = t;
    rec.p = intersection;
    rec.mat_ptr = this->mat;
    rec.set_face_normal(r, this->normal);

    return true;
}

double quad::get_D() const { return this->D; }
point3 quad::get_Q() const { return this->Q; }
vec3 quad::get_u() const { return this->u; }
vec3 quad::get_v() const { return this->v; }
vec3 quad::get_w() const { return this->w; }
vec3 quad::get_normal() const { return this->normal; }
shared_ptr<material> quad::get_mat() const { return this->mat; }
aabb quad::bounding_box() const { return this->bbox; }
