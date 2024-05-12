#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "geometry.h"

// PRIMITIVE INTERFACE
// Primitives are simple Geometry, usually requiring a small amount of instantiated RTCGeometry and materials.
// Parent to: SpherePrimitive

struct Vertex3f { float x, y, z; };
struct Quad { int v0, v1, v2, v3; };

class Primitive : public Geometry {
    public:
    shared_ptr<material> mat_ptr;


    Primitive(vec3 position, shared_ptr<material> m, RTCGeometry geom);
};

#endif