#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "geometry.h"

class Primitive : public Geometry {
    public:
    RTCGeometry geom;
    shared_ptr<material> mat_ptr;


    Primitive(vec3 position, shared_ptr<material> m, RTCGeometry geom) : geom{geom}, mat_ptr{m}, Geometry(position) {}
};

#endif