#include "scene.h"
#include <embree4/rtcore.h>

RTCScene initializeScene(RTCDevice device) {
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vertices = (float*) rtcSetNewGeometryBuffer(geom,
                                                     RTC_BUFFER_TYPE_VERTEX,
                                                     0,
                                                     RTC_FORMAT_FLOAT3,
                                                     3*sizeof(float),
                                                     3);

    unsigned* indices = (unsigned*) rtcSetNewGeometryBuffer(geom,
                                                          RTC_BUFFER_TYPE_INDEX,
                                                          0,
                                                          RTC_FORMAT_UINT3,
                                                          3*sizeof(unsigned),
                                                          1);
    // rtcSetNewGeometryBuffer creates data buffer. set of three vertices make a point, three points make a triangle. 
    // stored in 1D array, indices allocated to make it easy to access the values in the 1D array instead of 2D
    // assuming it's orderred like [x, ... ,x,y, ... ,y,z, ... ,z]
    // formula to get a certay x,y,z point is indices[point_ix] + indices.length()*dimension
    // for i >= 0, where point_ix is the intended 'point' from indices, and dimension ls 0,1,2, for x,y,z respectively

    if (vertices && indices) {
        vertices[0] = 0.f; vertices[1] = 0.f; vertices[2] = 0.f;
        vertices[3] = 1.f; vertices[4] = 0.f; vertices[5] = 0.f;
        vertices[6] = 0.f; vertices[7] = 1.f; vertices[8] = 0.f;

        indices[0] = 0; indices[1] = 1; indices[2] = 2;
    }

    // unsigned integer that retuns a geometry id
    // rtcAttachGeometry takes ownership of the geometry by increasing ref count thus we can release it
    rtcCommitGeometry(geom);
    unsigned int triangleID = rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);

    rtcCommitScene(scene);

    return scene;
}

void add_sphere(RTCDevice device, RTCScene scene) {
    
    RTCGeometry sphere = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
    float* spherev = (float*)rtcSetNewGeometryBuffer(sphere, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, 4*sizeof(float), 1);
    if (spherev) {
        spherev[0] = 0;
        spherev[1] = 0;
        spherev[2] = -1;
        spherev[3] = 0.5;
    }

    rtcCommitGeometry(sphere);
    unsigned int sphereID = rtcAttachGeometry(scene, sphere);
    rtcReleaseGeometry(sphere);
}

// add_sphere(scene,sphere_width,.....)