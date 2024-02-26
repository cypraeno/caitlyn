#ifndef QUAD_H
#define QUAD_H

#include "general.h"
#include "hittable.h"

class quad : public hittable {
    
    private:

        point3 Q;
        vec3 u, v;
        shared_ptr<material> mat;
        aabb bbox;
        vec3 normal;
        vec3 w;
        double D;

    public:

        quad(const point3& _Q, const vec3& _u, const vec3& _v, shared_ptr<material> _mat);

        virtual void set_bounding_box();

        bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

        // getter functions
        double get_D() const;
        point3 get_Q() const;
        vec3 get_u() const;
        vec3 get_v() const;
        vec3 get_w() const;
        vec3 get_normal() const;
        shared_ptr<material> get_mat() const;
        aabb bounding_box() const override;

};

#endif
