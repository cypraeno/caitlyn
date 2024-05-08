#ifndef MEDIUM_H
#define MEDIUM_H

#include "general.h"
#include "material.h"

class Medium {
    public:
    Medium(float density, shared_ptr<material> phase_function);

    virtual float particleDistance() const;

    /**
     * @brief calculates transmittance coefficient between point x and y, assuming a constant medium.
     * @note Assumes density as equivalent to extinction coefficient.
     * May not be accurate!
    */
    virtual float transmittance(const vec3& x, const vec3& y) const;
    virtual float transmittance(float dist) const;

    float density;
    shared_ptr<material> phase;
};


#endif