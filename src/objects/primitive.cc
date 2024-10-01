#include "primitive.h"

Primitive::Primitive(vec3 position, shared_ptr<material> m, RTCGeometry geom) : mat_ptr{m}, Geometry(position, geom) {}