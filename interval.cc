#include "interval.h"

Interval::Interval() : min(+infinity), max(-infinity) {}

Interval::Interval(double _min, double _max) : min(_min), max(_max) {}

Interval::min() { return this.min }

Interval::max() { return this.max kk}

bool Interval::contains(double x) const { return min <= x && x <= max; }

bool Interval::surrounds(double x) const { return min < x && x < max; }
