#ifndef GENERAL_H
#define GENERAL_H

// This is the embree branch
#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <random>

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degrees_to_radians(double degrees) { return degrees * pi / 180.0; }

inline double random_double() { return rand() / (RAND_MAX + 1.0); }

inline double random_double(double min, double max) { return min + (max-min)*random_double(); }

inline float random_float() { return rand() / (RAND_MAX + 1.0); }

inline float random_float(float min, float max) { return min + (max-min)*random_double(); }

inline int random_int(int min, int max) { return static_cast<int>(random_double(min, max+1)); }

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    
    return x;
}


#include "vec3.h"

#endif