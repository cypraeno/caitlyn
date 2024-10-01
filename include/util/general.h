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
const double euler = std::exp(1.0);
const double pi = 3.1415926535897932385;

double degrees_to_radians(double degrees);
double clamp(double x, double min, double max);

double random_double();
double random_double(double min, double max);
float random_float();
float random_float(float min, float max);
int random_int(int min, int max);

#include "vec3.h"

#endif