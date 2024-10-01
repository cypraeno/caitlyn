#include "geometry.h"

Geometry::Geometry(vec3 position, RTCGeometry geom) : geom{geom}, Visual(position) {}

Geometry::Geometry(shared_ptr<Geometry> geom_ptr) : Visual(geom_ptr->position) {
    geom = geom_ptr->geom;
    rtcRetainGeometry(geom);
}