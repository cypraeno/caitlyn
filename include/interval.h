#ifndef INTERVAL_H
#define INTERVAL_H

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

        bool contains(double x) const { return this->min <= x && x <= this->max; }

        bool surrounds(double x) const { return this->min < x && x < this->max; }

        inline double clamp(double x) const {
            if (x < this->min) return this->min;
            if (x > this->max) return this->max;
            return x;
        }

        static const interval empty, universe;
};

const static interval empty (+infinity, -infinity);
const static interval universe (-infinity, +infinity);

#endif

