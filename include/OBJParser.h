#ifndef OBJPARSER_H
#define OBJPARSER_H

#include "MTLParser.h"

/**
 * @brief Loads OBJ files into vectors of vertices, faces (containing normals, vertices, matkeys).
 * Uses MTL parser, so supports a limited range of MTL files. Does not support textures,
 * meaning that it will ignore all "vt" lines. 
*/
class OBJParser {
    public:
    bool parse(std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            return false;
        }

        size_t current_mat_idx = -1;

        std::string line;
        while (getline(file, line)) {
            std::istringstream lineStream(line);
            std::string lineType;
            lineStream >> lineType;

            if (lineType == "v") {
                float x, y, z;
                lineStream >> x >> y >> z;
                vec3 vertex(x, y, z);
                vertices.push_back(vertex);
            } else if (lineType == "f") {
                std::vector<int> face;

                std::string vertexData;

                // Push material index
                if (current_mat_idx == -1) { // no MTL file was ever defined
                    auto white = std::make_shared<lambertian>(color(0.73, 0.73, 0.73));
                    materials.push_back(white);
                    current_mat_idx = 0;
                }
                face.push_back(current_mat_idx);
                
                while (lineStream >> vertexData) {
                    std::istringstream vertexStream(vertexData);
                    std::string part;
                    int vertexIndex = -1, textureIndex = -1, normalIndex = -1;
                    int partIndex = 0;

                    while (getline(vertexStream, part, '/')) {
                        if (!part.empty()) { // Check if the part is not empty.
                            switch (partIndex) {
                                case 0: vertexIndex = std::stoi(part); break; // First part, vertex index.
                                case 1: textureIndex = std::stoi(part); break; // Second part, texture index.
                                case 2: normalIndex = std::stoi(part); break; // Third part, normal index.
                            }
                        }
                        partIndex++; // Move to the next part.
                    }
                    vertexIndex--; textureIndex--; normalIndex--;
                    
                    if (face.size() == 1) {
                        face.push_back(normalIndex);
                    }
                    face.push_back(vertexIndex);
                }

                if (face[1] < 0) { // normal not defined
                    // Calculates the face normal, pushes to array of normals and places the index in face.
                    int normal_idx;
                    normal_idx << normals.size();
                    vec3 v1 = vertices[face[2]] - vertices[face[3]];
                    vec3 v2 = vertices[face[2]] - vertices[face[4]];
                    vec3 normal = cross(v1, v2).unit_vector();
                    normals.push_back(normal);
                    face[1] = normal_idx;
                }
                faces.push_back(face);
            } else if (lineType == "vn") {
                float x, y, z;
                lineStream >> x >> y >> z;
                vec3 n(x, y, z);
                normals.push_back(n);
            } else if (lineType == "usemtl") {
                // Use MTL parser
                std::string id;
                lineStream >> id;

                current_mat_idx = matKeys[id];
            } else if (lineType == "mtllib") {
                std::string id;
                lineStream >> id;

                MTLParser mp;
                if (!mp.parse(id)) {
                    throw std::runtime_error("Failed to load .mtl file");
                }

                materials = mp.getMaterials();
                matKeys = mp.getMatKeys();
            }
        }

        return true;
    }

    const std::vector<vec3>& getVertices() const {
        return vertices;
    }

    const std::vector<std::vector<int>>& getFaces() const {
        return faces;
    }

    const std::vector<vec3>& getNormals() const {
        return normals;
    }

    const std::vector<shared_ptr<material>>& getMaterials() const {
        return materials;
    }

    private:
    std::vector<vec3> vertices;

    // {material_index, normal_index, v1_index, v2_index, ...}
    std::vector<std::vector<int>> faces;


    std::vector<vec3> normals;
    std::vector<shared_ptr<material>> materials;
    std::map<std::string, int> matKeys;
};

#endif