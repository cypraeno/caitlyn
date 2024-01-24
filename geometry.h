#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "visual.h"
#include "material.h"
#include <embree4/rtcore.h>

class Geometry : public Visual {
    public:
    Geometry(vec3 position) : Visual(position) {}

    virtual shared_ptr<material> materialById(unsigned int geomID) const = 0;
};

#endif