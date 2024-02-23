#include "hittable_list.h"

/* inline shared_ptr<hittable_list> box(const point3& a, const point3& b, shared_ptr<material> mat) {
    // Ensure a and b represent opposite corners of the box
    auto min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    auto max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

    // Calculate side vectors
    vec3 dx = max - min;
    vec3 dy(0, dx.y(), 0);
    vec3 dz(0, 0, dx.z());

    auto sides = make_shared<hittable_list>();
    
    // Create sides of the box
    auto add_side = [&](const point3& p, const vec3& u, const vec3& v) {
        sides->add(make_shared<quad>(p, u, v, mat));
    };

    add_side(min, dx, dy);                 // bottom
    add_side(min + dz, dx, dy);            // front
    add_side(min + dx + dz, -dz, dy);      // right
    add_side(min + dx, -dx, dy);           // back
    add_side(min + dx, -dx, -dz);          // left
    add_side(max, dx, -dz);                // top

    return sides;
}
 */