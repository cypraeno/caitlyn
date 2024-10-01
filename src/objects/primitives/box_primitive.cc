#include "box_primitive.h"

BoxPrimitive::BoxPrimitive(const point3& position, const vec3& a, const vec3& b, const vec3& c, std::shared_ptr<material> mat_ptr, RTCDevice device)
    : a{a}, b{b}, c{c}, Primitive{position, mat_ptr, rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD)} {

    // Assign vertices based on the position and directional vectors a, b, c
    const vec3 corners[] = {
        position,                                       // vertex 0
        position + a,                                   // vertex 1
        position + b,                                   // vertex 2
        position + a + b,                               // vertex 3
        position + c,                                   // vertex 4
        position + a + c,                               // vertex 5
        position + b + c,                               // vertex 6
        position + a + b + c                            // vertex 7
    };

    unsigned int indices[] = {
        0, 1, 3, 2, // front face
        5, 4, 6, 7, // back face
        0, 4, 5, 1, // bottom face
        2, 3, 7, 6, // top face
        0, 2, 6, 4, // left face
        1, 5, 7, 3  // right face
    };

    Vertex3f* vertices = (Vertex3f*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex3f), 8);
    for (int i = 0; i < 8; ++i) {  // Populate vertex buffer
        vertices[i].x = corners[i].x();
        vertices[i].y = corners[i].y();
        vertices[i].z = corners[i].z();
    }
    Quad* quads = (Quad*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(Quad), 6);
    for (int i = 0; i < 6; ++i) {
        quads[i].v0 = indices[4*i];
        quads[i].v1 = indices[4*i + 1];
        quads[i].v2 = indices[4*i + 2];
        quads[i].v3 = indices[4*i + 3];
    }
    
    rtcCommitGeometry(geom);
}

shared_ptr<material> BoxPrimitive::materialById(unsigned int geomID) const { return this->mat_ptr; }

HitInfo BoxPrimitive::getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const {
    HitInfo record;
    record.pos = p;
    record.t = t;
    
    // Calculate outward normal
    vec3 rel_pos = p - position;
    vec3 abs_a = a.unit_vector();
    vec3 abs_b = b.unit_vector();
    vec3 abs_c = c.unit_vector();
    float dp_a = dot(rel_pos, abs_a);
    float dp_b = dot(rel_pos, abs_b);
    float dp_c = dot(rel_pos, abs_c);
    vec3 normal;
    if (fabs(dp_a - a.length()) < 0.001) {
        normal = abs_a; // hit on face parallel to b-c plane, a direction
        record.u = dot(rel_pos, abs_b) / b.length();
        record.v = dot(rel_pos, abs_c) / c.length();
    } else if (fabs(dp_a) < 0.001) {
        normal = -abs_a;
        record.u = dot(rel_pos, abs_b) / b.length();
        record.v = dot(rel_pos, abs_c) / c.length();
    } else if (fabs(dp_b - b.length()) < 0.001) {
        normal = abs_b; // hit on face parallel to a-c plane, b direction
        record.u = dot(rel_pos, abs_a) / a.length();
        record.v = dot(rel_pos, abs_c) / c.length();
    } else if (fabs(dp_b) < 0.001) {
        normal = -abs_b;
        record.u = dot(rel_pos, abs_a) / a.length();
        record.v = dot(rel_pos, abs_c) / c.length();
    } else if (fabs(dp_c - c.length()) < 0.001) {
        normal = abs_c; // hit on face parallel to a-b plane, c direction
        record.u = dot(rel_pos, abs_a) / a.length();
        record.v = dot(rel_pos, abs_b) / b.length();
    } else if (fabs(dp_c) < 0.001) {
        normal = -abs_c;
        record.u = dot(rel_pos, abs_a) / a.length();
        record.v = dot(rel_pos, abs_b) / b.length();
    }
    record.set_face_normal(r, normal);
    return record;
}

vec3 BoxPrimitive::getA() { return a; }
vec3 BoxPrimitive::getB() { return b; }
vec3 BoxPrimitive::getC() { return c; }
