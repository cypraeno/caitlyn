#ifndef INTERSECTS_H
#define INTERSECTS_H

#include "ray.h"
#include <embree4/rtcore.h>

// Semi-temporary helper header file for the rtcIntersectX functions.
// Helpers do not actually fire the ray, they just set up the RTCRayHit objects with rays.

/** @brief modifies given RTCRayHit object to be ready for rtcIntersect1 usage */
void setupRayHit1(struct RTCRayHit& rayhit, const ray& r);

/** @brief modifies given RTCRayHit object to be ready for rtcIntersect4 usage*/
void setupRayHit4(struct RTCRayHit4& rayhit, std::vector<ray>& rays);

/** @brief modifies given RTCRayHit object to be ready for rtcIntersect8 usage*/
void setupRayHit8(struct RTCRayHit8& rayhit, std::vector<ray>& rays);

/** @brief modifies given RTCRayHit object to be ready for rtcIntersect16 usage*/
void setupRayHit16(struct RTCRayHit16& rayhit, std::vector<ray>& rays);

/**
 * @brief In a given scene, fires a continuous ray and fills information for multiple hits.
*/
void MultiIntersect(int max, ray r_in, RTCScene& rtc_scene, std::vector<int>& ids, std::vector<float>& tfars);

#endif