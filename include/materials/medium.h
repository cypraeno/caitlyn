#ifndef MEDIUM_H
#define MEDIUM_H

#include <algorithm>
#include <map>
#include "general.h"
#include "material.h"

class Medium {
    public:
    Medium(float density, shared_ptr<material> phase_function);

    virtual float particleDistance() const;

    /**
     * @brief calculates transmittance coefficient between point x and y, assuming a constant medium.
     * @note Assumes density as equivalent to extinction coefficient.
     * May not be accurate!
    */
    virtual float transmittance(const vec3& x, const vec3& y) const;
    virtual float transmittance(float dist) const;

    float density;
    shared_ptr<material> phase;
};


/**
 * @class MediumRecord
 * @brief Used for two purposes: tracking of mediums when ray tracing/marching and for calculating transmittance in DLS.
*/
class MediumRecord {
    public:
    std::vector<shared_ptr<Medium>> mediums;
    
    // Transmittance calculation variables
    float transmittance = 1.0;
    point3 recent_position;
    shared_ptr<Medium> highest_density_volume = nullptr;



    MediumRecord(point3 initial_position);
    
    bool contains(std::shared_ptr<Medium> vol) const;
    
    /**
     * @brief Used to update the mediums list and update transmittance.
     * Called whenever a new volume is hit in which we travel through the participating mediums.
     * 
     * 1. Let D be the distance between new point and previous point. This is distance traveled.
     * 2. Multiply transmittance by the transmittance applied by the highest_density_volume in D distance.
     * 3A. If volume is in mediums, then we are exiting - remove from mediums.
     * 3B. If volume is NOT in mediums, then we are entering. Check if vol->density > highest_density_volume->density.
    */
    void hitVolume(shared_ptr<Medium> vol, point3 hitPoint);

    /**
     * @brief Only used for direct light sampling. Given
    */

    /**
     * @brief Iterates through mediums and calls particleDistance to simulate the nearest particle of all
     * participating mediums.
     * @returns ptr to the particle's corresponding medium and updates passed in dist float.
    */
    shared_ptr<Medium> particleDistance(float& dist) const;

    
};


#endif