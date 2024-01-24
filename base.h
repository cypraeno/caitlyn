#ifndef BASE_H
#define BASE_H

#include "vec3.h"

// MESH INTERFACE
// base class for all 3d objects. Contains only two properties.

class Base {
    public:
    vec3 position;
    bool active = true;

    Base(vec3 position) : position{position} {}
}


#endif