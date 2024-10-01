#include "intersects.h"

void setupRayHit1(struct RTCRayHit& rayhit, const ray& r) {
    rayhit.ray.org_x = r.origin().x();
    rayhit.ray.org_y = r.origin().y();
    rayhit.ray.org_z = r.origin().z();
    rayhit.ray.dir_x = r.direction().x();
    rayhit.ray.dir_y = r.direction().y();
    rayhit.ray.dir_z = r.direction().z();
    rayhit.ray.tnear = 0.001;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
}

void setupRayHit4(struct RTCRayHit4& rayhit, std::vector<ray>& rays) {
    int ix = 0;
    for(auto r: rays) {
        rayhit.ray.org_x[ix] = r.origin().x();
        rayhit.ray.org_y[ix] = r.origin().y();
        rayhit.ray.org_z[ix] = r.origin().z();
        rayhit.ray.dir_x[ix] = r.direction().x();
        rayhit.ray.dir_y[ix] = r.direction().y();
        rayhit.ray.dir_z[ix] = r.direction().z();
        rayhit.ray.tnear[ix] = 0.001;
        rayhit.ray.tfar[ix] = std::numeric_limits<float>::infinity();
        rayhit.ray.mask[ix] = -1;
        rayhit.ray.flags[ix] = 0;
        rayhit.hit.geomID[ix] = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0][ix] = RTC_INVALID_GEOMETRY_ID;
        ix += 1;
    }
}

void setupRayHit8(struct RTCRayHit8& rayhit, std::vector<ray>& rays) {
    int ix = 0;
    for(auto r: rays) {
        rayhit.ray.org_x[ix] = r.origin().x();
        rayhit.ray.org_y[ix] = r.origin().y();
        rayhit.ray.org_z[ix] = r.origin().z();
        rayhit.ray.dir_x[ix] = r.direction().x();
        rayhit.ray.dir_y[ix] = r.direction().y();
        rayhit.ray.dir_z[ix] = r.direction().z();
        rayhit.ray.tnear[ix] = 0.001;
        rayhit.ray.tfar[ix] = std::numeric_limits<float>::infinity();
        rayhit.ray.mask[ix] = -1;
        rayhit.ray.flags[ix] = 0;
        rayhit.hit.geomID[ix] = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0][ix] = RTC_INVALID_GEOMETRY_ID;
        ix += 1;
    }
}

void setupRayHit16(struct RTCRayHit16& rayhit, std::vector<ray>& rays) {
    int ix = 0;
    for(auto r: rays) {
        rayhit.ray.org_x[ix] = r.origin().x();
        rayhit.ray.org_y[ix] = r.origin().y();
        rayhit.ray.org_z[ix] = r.origin().z();
        rayhit.ray.dir_x[ix] = r.direction().x();
        rayhit.ray.dir_y[ix] = r.direction().y();
        rayhit.ray.dir_z[ix] = r.direction().z();
        rayhit.ray.tnear[ix] = 0.001;
        rayhit.ray.tfar[ix] = std::numeric_limits<float>::infinity();
        rayhit.ray.mask[ix] = -1;
        rayhit.ray.flags[ix] = 0;
        rayhit.hit.geomID[ix] = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0][ix] = RTC_INVALID_GEOMETRY_ID;
        ix += 1;
    }
}

void MultiIntersect(int max, ray r_in, RTCScene& rtc_scene, std::vector<int>& ids, std::vector<float>& tfars) {
    struct RTCRayHit rayhit;
    ray r = r_in;
    setupRayHit1(rayhit, r);
    for (int i=0; i<max; i++) {
        rtcIntersect1(rtc_scene, &rayhit);
        int targetID;
        if (rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID) {
            targetID = rayhit.hit.instID[0];
        } else if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            targetID = rayhit.hit.geomID;
        } else {
            break;
        }
        ids.push_back(targetID);
        // Calculate tfar for the original ray
        float time = (r.at(rayhit.ray.tfar) - r_in.origin()).length() / (r_in.direction().length());
        tfars.push_back(time);
        r = ray(r.at(rayhit.ray.tfar), r.direction(), 0.0);
        setupRayHit1(rayhit, r);
    }
}