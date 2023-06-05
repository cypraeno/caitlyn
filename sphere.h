#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"


/**
 * implementation of a sphere
 * 
 * the sphere inherits from the hittable class and is used to create spheres
 * which can be hit in a scene
*/
class sphere : public hittable {

    // constructors
    public:
        __device__ sphere() {}
        __device__ sphere(point3 cen, float r) : center(cen), radius(r) {};

    // methods
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;    // sphere hit?

    // attributes
    public:
        point3 center;
        float radius;
};

/**
 * determines if sphere was hit and records the hit if it was
 * 
 * @param[in] r the ray being shot out from the eye
 * @param[in] t_min, t_max the interval of acceptable intersections between the ray and sphere
 * @param[out] rec the details of the intersection
 * 
 * @return if the sphere was hit
 * 
 * @relatesalso sphere
*/
__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;

    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius*radius;

    float discriminant = half_b*half_b - a*c;

    if (discriminant > 0) {

        float sqrtd = sqrt(determinant);

        // nearest root in acceptable range
        float root = (-half_b - sqrtd) / a;
        if (t_min < root && root < t_max) {
            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            return true;
        }

        root = (-half_b + sqrtd) / a;
        if (t_min < root && root < t_max) {
            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            return true;
        }
    }
    
    return false;
}


#endif
