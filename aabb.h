#ifndef AABB_H
#define AABB_H

#include "ray.h"
#include "interval.h"

using std::fmin;
using std::fmax;

class aabb {
public:
    interval x, y, z;
    aabb() {} // note default ctor of interval sets min and max to 0

    aabb(const interval& ix, const interval& iy, const interval& iz): x(ix), y(iy), z(iz) {}

    aabb(const point3& a, const point3& b) {
        // set the AABB to the interval of the two points
        x = interval(fmin(a[0],b[0]), fmax(a[0],b[0]));
        y = interval(fmin(a[1],b[1]), fmax(a[1],b[1]));
        z = interval(fmin(a[2],b[2]), fmax(a[2],b[2]));
    }
    aabb(const aabb& box0, const aabb& box1) {
        x = interval(box0.x, box1.x);
        y = interval(box0.y, box1.y);
        z = interval(box0.z, box1.z);
    }

    const interval& axis(int n) const {
        if (n == 1) return y;
        if (n == 2) return z;
        return x;
    }

    bool hit(const ray& r, interval ray_t) const {

        for (int a = 0; a < 3; a++) {

            auto inv_direction = 1 / r.direction()[a];
            auto orig = r.origin()[a];

            auto t0 = (this->axis(a).min - orig) * inv_direction;
            auto t1 = (this->axis(a).max - orig) * inv_direction;

            if (inv_direction < 0) std::swap(t0, t1);

            if (t0 > ray_t.min) ray_t.min = t0;
            if (t1 < ray_t.max) ray_t.max = t1;

            if (ray_t.max <= ray_t.min) return false;
        }

        return true;
    }

    /**
     * @return aabb padded to prevent zero thickness
     */
    aabb pad() {
    double delta = 0.0001;
    interval new_x = (x.size() >= delta) ? x : x.expand(delta);
    interval new_y = (y.size() >= delta) ? y : y.expand(delta);
    interval new_z = (z.size() >= delta) ? z : z.expand(delta);

    return aabb(new_x, new_y, new_z);
}
};


#endif
