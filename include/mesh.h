#ifndef MESH_H
#define MESH_H

#include "geometry.h"
#include "quad_primitive.h"
#include "OBJParser.h"

class Mesh : public Geometry {
    public:
    // assuming geomIDs increment, if startId is 5, and geomID is 6, then
    // geomID 6 refers to -> face idx 1


    std::vector<RTCGeometry> geoms;
    shared_ptr<material> mat_ptr;

    Mesh(vec3 position, shared_ptr<material> mat_ptr, std::string& filePath, RTCDevice device) : mat_ptr{mat_ptr}, Geometry(position) {
        loadGeometry(filePath, device);
    }

    void loadGeometry(std::string& filePath, RTCDevice device) {
        OBJParser parser;
        if (!parser.parse(filePath)) {
            throw std::runtime_error("Failed to load .obj file");
        }

        vertices = parser.getVertices();
        faces = parser.getFaces();

        for (const auto& face : faces) {
            if (face.size() != 4) {
                continue;
            }

            RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD);
            Vertex3f* vertBuffer = (Vertex3f*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex3f), 4);
            for (size_t i = 0; i < face.size(); ++i) {
                vec3 vertice = vertices[face[i]];
                vertBuffer[i].x = vertice.x();
                vertBuffer[i].y = vertice.y();
                vertBuffer[i].z = vertice.z();
            }
            
            Quad* quad = (Quad*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(Quad), 1);
            quad[0].v0 = 0; quad[0].v1 = 1; quad[0].v2 = 2; quad[0].v3 = 3;

            rtcSetGeometryVertexAttributeCount(geom, 1);
            rtcSetGeometryBuildQuality(geom, RTC_BUILD_QUALITY_HIGH);
            rtcCommitGeometry(geom);
            geoms.push_back(geom);
        }
    }


    shared_ptr<material> materialById(unsigned int geomID) const override {
        return mat_ptr;
    }

    HitInfo getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const override {
        HitInfo record;

        int faces_idx = geomID - starterId;
        std::vector face = faces[faces_idx];
        vec3 v1 = vertices[face[1]] - vertices[face[0]];
        vec3 v2 = vertices[face[2]] - vertices[face[0]];

        vec3 normal = vec3(
            v1.y() * v2.z() - v1.z() * v2.y(),
            v1.z() * v2.x() - v1.x() * v2.z(),
            v1.x() * v2.y() - v1.y() * v2.x()
        ).unit_vector();

        record.pos = p;
        record.t = t;
        record.set_face_normal(r, normal);

        record.u = 0;
        record.v = 0;

        return record;
    }

    void setStarterId(unsigned int x) {
        starterId = x;
    }

    private:
    std::vector<vec3> vertices;
    std::vector<std::vector<int>> faces;
    unsigned int starterId;
};

#endif