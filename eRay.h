#ifndef ERAY_H
#define ERAY_H

#include <limits>
#include <embree4/rtcore.h>

#include "vec3.h"

using namespace std;

/** @brief implementation of a wrapper for struct RTCRay */
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
            unsigned int mask=0, unsigned int flags=0);
        
        point3 getOrg() const;              /**< returns ray org */
        vec3 getDir() const;                /**< returns ray dir */
        float getTime() const;              /**< returns ray time */
        float getTNear() const;             /**< returns ray tnear */
        float getTFar() const;              /**< returns ray tfar */
        unsigned int getID() const;         /**< returns ray ID */
        unsigned int getMask() const;       /**< returns ray mask */
        unsigned int getFlags() const;      /**< returns ray flags */

        /**  
         * @brief return the point on the ray at length t
         * 
         * @param[in] t the length of the array from org
         * 
         * @return Point3 obj of the point on the ray
         */
        point3 at(float t) const;

        /**
         * @brief creates RTCRay struct based on object data
         * 
         * @param[out] ray the struct to store ray data
         * 
         * @note existing RTCRay fields will be overwritten
        */
        void createRTCRay(struct RTCRay& ray) const;

};

#endif