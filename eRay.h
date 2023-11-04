#ifndef RAY_H
#define RAY_H

#include <limits>
#include <embree/rtcore.h>

#include "vec3.h"

using namespace std;

/** implementation of a wrapper for struct RTCRay */
class Ray {

    static unsigned int id;                 /**< unique ID to discern rays */
    point3 org;                             /**< Point3 object of the ray origin */
    vec3 dir;                               /**< Vec object of the ray direction (vec does not need to be normalized) */
    float time;                             /**< ray time (for motion blur) */
    float tnear, tfar;                      /**< ray interval (must be in range [0, ∞]; tfar >= tnear) */
    unsigned int mask;                      /**< RTCRay geometry mask */
    unsigned int flags;                     /**< RTCRay flags */

    public: 

        Ray(const point3& org, const vec3& dir, float time=0.0, 
            float tnear=0.0, float tfar=numeric_limits<float>::infinity(), 
            unsigned int mask, unsigned int flags=0);
        
        point3 getOrg() const;              /**< unique ID to discern rays */
        vec3 getDir() const;
        float getTime() const;
        float getTNear() const;
        float getTFar() const;
        unsigned int getID() const;
        unsigned int getMask() const;
        unsigned int getFlags() const;

        point3 at() const;

        void createRTCRay(struct RTCRay& rtcRay) const;




}

#endif