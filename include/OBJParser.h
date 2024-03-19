#ifndef OBJPARSER_H
#define OBJPARSER_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vec3.h"

class OBJParser {
    public:
    bool parse(std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            return false;
        }

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
                std::string vertexIndex;
                while (lineStream >> vertexIndex) {
                    size_t slashPos = vertexIndex.find('/');
                    int idx = std::stoi(vertexIndex.substr(0, slashPos)) - 1; // Convert 1-based index to 0-based
                    face.push_back(idx);
                }
                faces.push_back(face);
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

    private:
    std::vector<vec3> vertices;
    std::vector<std::vector<int>> faces;
};

#endif