#ifndef SCENE_H
#define SCENE_H

#include <embree4/rtcore.h>

RTCScene initializeScene(RTCDevice device);

void add_sphere(RTCDevice device, RTCScene scene);

#endif 