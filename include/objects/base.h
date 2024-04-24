#ifndef BASE_H
#define BASE_H

#include "general.h"

// MESH INTERFACE
// base class for all 3d objects. Contains only two properties.
// Parent to: Visual, Camera

/**
 * @class Base
 * @brief Represents a basic 3D object in a scene.
 * 
 * @param[in]       position
 */
class Base {
    public:
    vec3 position;
    bool active = true; // as of writing this, this property isn't actually used anywhere. There's no way to deactivate.

    Base(vec3 position);
};


#endif