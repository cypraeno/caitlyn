#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
    
    public:

        ray();
        ray(const point3& origin, const vec3& direction);
        ray(const point3& origin, const vec3& direction, double time = 0.0);

        point3 origin() const;
        vec3 direction() const;
        double time() const;

        point3 at(double t) const;

    private: 

        point3 orig;
        vec3 dir;
        double tm;
};

#endif