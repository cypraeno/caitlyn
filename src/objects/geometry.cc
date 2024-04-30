#include "geometry.h"

Geometry::Geometry(vec3 position, RTCGeometry geom) : geom{geom}, Visual(position) {}