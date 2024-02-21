#include "quad.h"

quad::quad(const point3& _Q, const vec3& _u, const vec3& _v, shared_ptr<material> _mat) :
    Q(_Q), u(_u), v(_v), mat(_mat) {
        this->set_bounding_box();
    }

virtual void quad::set_bounding_box() {
    this->bbox = aabb(Q, Q + u + v).pad();
}

point3 get_Q() const { return this->Q; }
vec3 get_u() const { return this->u; }
vec3 get_u() const { return this->v; }
shared_ptr<material> get_mat() const { return this->mat; }
aabb quad::bounding_box() const override { return this->bbox; }

bool quad::hit(const ray& r, interval ray_t, hit_record& rec) const override {
    return false; // To be implemented
}
