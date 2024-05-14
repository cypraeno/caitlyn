// general.cc

#include "general.h"

double degrees_to_radians(double degrees) { return degrees * pi / 180.0; }

double random_double() { return rand() / (RAND_MAX + 1.0); }

double random_double(double min, double max) { return min + (max-min)*random_double(); }

float random_float() { return rand() / (RAND_MAX + 1.0); }

float random_float(float min, float max) { return min + (max-min)*random_double(); }

int random_int(int min, int max) { return static_cast<int>(random_double(min, max+1)); }

double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    
    return x;
}
