#include "medium.h"

Medium::Medium(float density, shared_ptr<material> phase_function) 
    : density{density}, phase{phase_function} {}

float Medium::particleDistance() const {
    float neg_inv_density = -1 / density;
    auto hit_distance = neg_inv_density * log(random_double());
    return hit_distance;
}