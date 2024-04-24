#ifndef INTERSECTS_H
#define INTERSECTS_H

// Helper header file for the rtcIntersectX functions.
// Helpers do not actually fire the ray, they just set up the RTCRayHit objects with rays.

/** @brief modifies given RTCRayHit object to be ready for rtcIntersect1 usage */
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

/** @brief modifies given RTCRayHit object to be ready for rtcIntersect4 usage*/
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

/** @brief modifies given RTCRayHit object to be ready for rtcIntersect8 usage*/
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

/** @brief modifies given RTCRayHit object to be ready for rtcIntersect16 usage*/
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

#endif