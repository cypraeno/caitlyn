#ifndef GENERAL_H
#define GENERAL_H

#include <cmath>
#include <limits>
#include <memory>


using std::shared_ptr;
using std::make_shared;
using std::sqrt;

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

#include "ray.h"
#include "vec3.h"
#include <random>

inline double random_double() { // newer "better?" implementation using the <random> module
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}
// Added this function in for cases where range is not [0, 1)
inline double random_double(double min, double max ) {
    static std::uniform_real_distribution<double> distribution(min, max);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}


#endif