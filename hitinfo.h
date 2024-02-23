#ifndef HITINFO_H
#define HITINFO_H

struct HitInfo {
    point3 pos;
    vec3 normal;
    bool front_face;
    float t;
    double u;
    double v;

    /** @brief Given a face's outward normal and the initial ray, sets front_face to represent
    if collision hits it from the front or not. */
    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


#endif