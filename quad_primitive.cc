#include "quad_primitive.h"
#include <stdio.h>

QuadPrimitive::QuadPrimitive(const point3& position, const vec3& _u, const vec3& _v, shared_ptr<material> mat_ptr, RTCDevice device):
    u{_u}, v{_v}, Primitive{position, mat_ptr, rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD)} {

    vec3 n = cross(this->u,this->v);
    this->normal = n.unit_vector();
    this->w = n / dot(n, n);
    vec3 pu = this->position + this->u;
    vec3 pv = this->position + this->v;
    vec3 uv = this->position + this->u + this->v;

    Vertex3f* quadv = (Vertex3f*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex3f), 4);
    quadv[0].x = this->position.x();    quadv[0].y = this->position.y();    quadv[0].z = this->position.z();      // p0
    quadv[1].x = pu.x();                quadv[1].y = pu.y();                quadv[1].z = pu.z();                  // p1
    quadv[2].x = uv.x();                quadv[2].y = uv.y();                quadv[2].z = uv.z();                  // p2
    quadv[3].x = pv.x();                quadv[3].y = pv.y();                quadv[3].z = pv.z();                  // p3

    Quad* quad = (Quad*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(Quad), 1);
    quad[0].v0 = 0; quad[0].v1 = 1; quad[0].v2 = 2; quad[0].v3 = 3;

    rtcSetGeometryVertexAttributeCount(geom, 1);

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
