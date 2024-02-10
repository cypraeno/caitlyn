#ifndef INTERVAL_H
#define INTERVAL_H

#include "vec3.h"


using std::fmax;
using std::fmin;

class interval {
public:
    double min;
    double max;
    interval(): min(0), max(0) {}
    interval(double min, double max): min(min), max(max) {}
    interval(const interval& a, const interval& b): min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

    double size() const { return max - min; }

    interval expand(double delta) const {
        auto padding = delta / 2;
        return interval(min - padding, max + padding);
    }
    
};


#endif

