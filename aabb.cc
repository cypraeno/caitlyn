#include "aabb.h"

aabb::pad() {
    double delta = 0.0001;
    interval new_x = (x.size() >= delta) ? x : x.expand(delta);
    interval new_y = (y.size() >= delta) ? y : y.expand(delta);
    interval new_z = (z.size() >= delta) ? z : z.expand(delta);

    return aabb(new_x, new_y, new_z);
}
