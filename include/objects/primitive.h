#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "geometry.h"

// PRIMITIVE INTERFACE
// Primitives are simple Geometry, usually requiring a small amount of instantiated RTCGeometry and materials.
// Parent to: SpherePrimitive

class Primitive : public Geometry {
    public:
    RTCGeometry geom;
    shared_ptr<material> mat_ptr;


    Primitive(vec3 position, shared_ptr<material> m, RTCGeometry geom);
};

#endif