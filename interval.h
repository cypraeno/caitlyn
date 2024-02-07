#ifndef INTERVAL_H
#define INTERVAL_H

class Interval {

    public:
        double min, max;

        Interval();
        Interval(double _min, double _max);

        min();
        max();

        bool contains(double x) const;

        bool surrounds(double x) const;

        double clamp(double x) const;

        static const Interval empty, universe;
};

const static interval empty (+infinity, -infinity);
const static interval universe (-infinity, +infinity);

#endif
