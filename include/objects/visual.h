#ifndef VISUAL_H
#define VISUAL_H

#include "base.h"

// VISUAL (obj) INTERFACE
// represents all objects whose existence affects, visually, the end render.
// Includes:
// => Lights
// => Geometry

// Parent to: Geometry

// In the future, this layer exists for adding optimization with AABBs.

class Visual : public Base {
    public:
    Visual(vec3 position);
};

#endif