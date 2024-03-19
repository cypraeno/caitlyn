#ifndef OBJPARSER_H
#define OBJPARSER_H

#include "MTLParser.h"

class OBJParser {
    public:
    bool parse(std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            return false;
        }

        size_t current_mat_idx = 0;

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

                // Push material index
                face.push_back(current_mat_idx);

                std::string element;
                while (lineStream >> element) {

                    std::replace(element.begin(), element.end(), '/', ' ');
                    std::istringstream vertexStream(element);
                    int vertexIndex, textureIndex, normalIndex;
                    vertexStream >> vertexIndex >> textureIndex >> normalIndex;
                    vertexIndex--; textureIndex--; normalIndex--;

                    if (face.size() == 1) { // first vertex being parsed
                        face.push_back(normalIndex);
                    }
                    face.push_back(vertexIndex);
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