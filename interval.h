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
    interval(double _min, double _max): min(_min), max(_max) {}
    interval(const interval& a, const interval& b): min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

    double size() const { return max - min; }

    interval expand(double delta) const {
        auto padding = delta / 2;
        return interval(min - padding, max + padding);
    }

    bool contains(double x) const { return this.min <= x && x <= this.max}

    bool surrounds(double x) const { return this.min < x && x < this.max }

    static const interval empty, universe;
};

const static interval empty (+infinity, -infinity);
ocnst static interval universe (-infinity, +infinity);

#endif

