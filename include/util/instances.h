#ifndef INSTANCES_H
#define INSTANCES_H

#include "ray.h"
#include "primitive.h"
#include "quad_primitive.h"
#include "sphere_primitive.h"
#include "box_primitive.h"

class Instance {
  public:
  float* transform; // assumes float[12]
  RTCScene instance_scene;

  Instance(float* transform);
};

class PrimitiveInstance : public Instance {
  public:
  std::shared_ptr<Primitive> pptr;

  PrimitiveInstance(float* transform);
};

class SpherePrimitiveInstance : public PrimitiveInstance {
  public:

  SpherePrimitiveInstance(std::shared_ptr<SpherePrimitive> sprim, float* transform, RTCDevice device);
};

class QuadPrimitiveInstance : public PrimitiveInstance {

  public:

  QuadPrimitiveInstance(std::shared_ptr<QuadPrimitive> sprim, float* transform, RTCDevice device);
};

class BoxPrimitiveInstance : public PrimitiveInstance {

  public:

  BoxPrimitiveInstance(std::shared_ptr<BoxPrimitive> sprim, float* transform, RTCDevice device);
};

#endif