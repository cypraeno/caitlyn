#include "medium.h"

Medium::Medium(float density, shared_ptr<material> phase_function) 
    : density{density}, phase{phase_function} {}

float Medium::particleDistance() const {
    float neg_inv_density = -1 / density;
    auto hit_distance = neg_inv_density * log(random_double());
    return hit_distance;
}

float Medium::transmittance(const vec3& x, const vec3& y) const {
    float exponent = -(density * fabs((x - y).length()));
    return pow(euler, exponent);
}

float Medium::transmittance(float dist) const {
    float exponent = -(density * dist);
    return pow(euler, exponent);
}


// MediumStack

MediumRecord::MediumRecord(point3 initial_position) : recent_position{initial_position} {}

bool MediumRecord::contains(std::shared_ptr<Medium> vol) const {
    return std::find(mediums.begin(), mediums.end(), vol) != mediums.end();
}

void MediumRecord::hitVolume(shared_ptr<Medium> vol, point3 hitPoint) {
    float distance_traveled = (recent_position - hitPoint).length();
    if (!mediums.empty()) {
        transmittance *= highest_density_volume->transmittance(distance_traveled);
        if (!vol) { return; }
        if (!contains(vol)) { // entering
            mediums.push_back(vol);
            if (vol->density > highest_density_volume->density) { highest_density_volume = vol; }
        } else { // exiting
            mediums.erase(std::remove(mediums.begin(), mediums.end(), vol), mediums.end());

            float highestDensity = -1.0f;
            if (vol == highest_density_volume) { // reset highest density volume by finding it
                for (const auto& ptr : mediums) {
                    if (ptr->density > highestDensity) {
                        highestDensity = ptr->density;
                        highest_density_volume = ptr;
                    }
                }
            }
        }
    } else { // mediums empty
        if (!vol) { return; }
        highest_density_volume = vol;
        mediums.push_back(vol);
        recent_position = hitPoint;
    }
}

shared_ptr<Medium> MediumRecord::particleDistance(float& dist) const {
    float min_dist;
    shared_ptr<Medium> p = nullptr;
    for (size_t i = 0; i < mediums.size(); ++i) {
        float ndist = mediums[i]->particleDistance();
        if (i == 0) {
            min_dist = ndist;
            p = mediums[i];
        } else {
            if (ndist < min_dist) {
                min_dist = ndist;
                p = mediums[i];
            }
        }
    }
    dist = min_dist;
    return p;
}