#include "quad_primitive.h"

QuadPrimitive::QuadPrimitive(const point3& position, const vec3& _u, const vec3& _v, shared_ptr<material> mat_ptr, RTCDevice device):
    u{_u}, v{_v}, Primitive{position, mat_ptr, rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD)} {

    vec3 n = cross(u, v);
    this->normal = n.unit_vector();
    this->w = n / dot(n, n);

    float* quadv = (float*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3*sizeof(float), 4);
    // if memory allocation succeeded
    if (quadv) {
        quadv[0] = this->position.x();      quadv[1]  = this->position.y();      quadv[2]  = this->position.z();        // p0
        quadv[3] = this->u.x();             quadv[4]  = this->u.y();             quadv[5]  = this->u.z();               // p1
        quadv[6] = (this->u + this->v).x(); quadv[7]  = (this->u + this->v).y(); quadv[8]  = (this->u + this->v).z();   // p2
        quadv[9] = this->v.x();             quadv[10] = this->v.y();             quadv[11] = this->v.z();               // p3
    }

    rtcSetGeometryBuildQuality(geom, RTC_BUILD_QUALITY_HIGH);
    rtcCommitGeometry(geom);
}

shared_ptr<material> QuadPrimitive::materialById(unsigned int geomID) const { return this->mat_ptr; }

HitInfo QuadPrimitive::getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const {
    HitInfo record;
    record.pos = p;
    record.t = t;
    record.set_face_normal(r, this->normal);

    vec3 planar_hitpoint = p - this->position;
    record.u = dot(this->w, cross(planar_hitpoint, this->v));
    record.v = dot(this->w, cross(this->u, planar_hitpoint));

    return record;
}
