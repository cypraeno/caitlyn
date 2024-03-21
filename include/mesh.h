#ifndef MESH_H
#define MESH_H

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include "geometry.h"
#include "quad_primitive.h"
#include "OBJParser.h"

/**
 * @brief Mesh object constructed with a given OBJ with MTL file.
 * Loads into a series of RTCGeometry geoms, normals, and materials.
 * Uses a "starterId" to convert geomIDs to real indexes to the vectors.
*/
class Mesh : public Geometry {
    public:
    // assuming geomIDs increment, if startId is 5, and geomID is 6, then
    // geomID 6 refers to -> face idx 1


    std::vector<RTCGeometry> geoms;
    float scale;

    Mesh(vec3 position, float scale, std::string& filePath, RTCDevice device) : scale{scale}, Geometry(position) {
        loadGeometry(filePath, device);
    }

    /**
     * @brief loadGeometry takes in a path to an obj file, uses a parseer to load it, and constructs
     * vector of RTCGeometry objects based on the list of originalFaces and vertices.
     * loadGeometry will use triangles and quads for 3 and 4 vertice originalFaces respectively.
     * For originalFaces with > 4 vertices, it will create a triangle fan.
    */
    void loadGeometry(std::string& filePath, RTCDevice device) {
        OBJParser parser;
        if (!parser.parse(filePath)) {
            throw std::runtime_error("Failed to load .obj file");
        }

        vertices = parser.getVertices();
        originalFaces = parser.getFaces();
        normals = parser.getNormals();
        materials = parser.getMaterials();

        for (const auto& face : originalFaces) { // Use a reference to avoid copying here
            size_t numVertices = face.size() - 2;

            if (numVertices == 3) { // triangle
                RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
                Vertex3f* vertBuffer = (Vertex3f*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex3f), 3);
                for (size_t i = 0; i < 3; ++i) {
                    vec3 vertice = vertices[face[i + 2]];
                    vertBuffer[i].x = vertice.x() * scale + position.x();
                    vertBuffer[i].y = vertice.y() * scale + position.y();
                    vertBuffer[i].z = vertice.z() * scale + position.z();
                }
                unsigned* indexBuffer = (unsigned*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(unsigned) * 3, 1);
                indexBuffer[0] = 0; indexBuffer[1] = 1; indexBuffer[2] = 2;

                rtcCommitGeometry(geom);
                geoms.push_back(geom);
                faces.push_back(face);
            } else if (numVertices == 4) {
                RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD);
                Vertex3f* vertBuffer = (Vertex3f*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex3f), 4);
                for (size_t i = 0; i < numVertices; ++i) {
                    vec3 vertice = vertices[face[i + 2]];
                    vertBuffer[i].x = vertice.x() * scale + position.x();
                    vertBuffer[i].y = vertice.y() * scale + position.y();
                    vertBuffer[i].z = vertice.z() * scale + position.z();
                }
                
                Quad* quad = (Quad*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(Quad), 1);
                quad[0].v0 = 0; quad[0].v1 = 1; quad[0].v2 = 2; quad[0].v3 = 3;

                rtcCommitGeometry(geom);
                geoms.push_back(geom);
                faces.push_back(face);
            } else {
                for (size_t i = 1; i < numVertices - 1; ++i) {
                    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
                    Vertex3f* vertBuffer = (Vertex3f*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex3f), 3);

                    vec3 vertice0 = vertices[face[2]];
                    vertBuffer[0].x = vertice0.x() * scale + position.x();
                    vertBuffer[0].y = vertice0.y() * scale + position.y();
                    vertBuffer[0].z = vertice0.z() * scale + position.z();

                    vec3 vertice1 = vertices[face[i + 2]];
                    vec3 vertice2 = vertices[face[i + 2 + 1]];
                    vertBuffer[1].x = vertice1.x() * scale + position.x();
                    vertBuffer[1].y = vertice1.y() * scale + position.y();
                    vertBuffer[1].z = vertice1.z() * scale + position.z();
                    vertBuffer[2].x = vertice2.x() * scale + position.x();
                    vertBuffer[2].y = vertice2.y() * scale + position.y();
                    vertBuffer[2].z = vertice2.z() * scale + position.z();

                    unsigned* indexBuffer = (unsigned*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(unsigned) * 3, 1);
                    indexBuffer[0] = 0; indexBuffer[1] = 1; indexBuffer[2] = 2;

                    rtcCommitGeometry(geom);
                    geoms.push_back(geom);
                    faces.push_back({face[0], face[i], face[i + 1]});
                }
            }

            //rtcSetGeometryVertexAttributeCount(geom, 1);
            //rtcSetGeometryBuildQuality(geom, RTC_BUILD_QUALITY_HIGH);
        }
    }


    shared_ptr<material> materialById(unsigned int geomID) const override {
        int faces_idx = geomID - starterId;
        std::vector<int> face = faces[faces_idx];

        return materials[face[0]];
    }

    HitInfo getHitInfo(const ray& r, const vec3& p, const float t, unsigned int geomID) const override {
        HitInfo record;
        int faces_idx = geomID - starterId;
        std::vector<int> face = faces[faces_idx];
        size_t numVertices = face.size();
        
        vec3 normal = normals[face[1]];

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
    std::vector<std::vector<int>> originalFaces;
    std::vector<std::vector<int>> faces;
    std::vector<vec3> normals;
    std::vector<shared_ptr<material>> materials;
    unsigned int starterId;
};

#endif