#ifndef MEDIUM_H
#define MEDIUM_H

#include "material.h"

class Medium {
    public:
    Medium(float density, shared_ptr<material> phase_function);

    virtual float particleDistance() const;

    float density;
    shared_ptr<material> phase;
};


#endif