// Semi-temporary helper header file for the rtcIntersectX functions.
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