#ifndef MTLPARSER_H
#define MTLPARSER_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <map>

#include "vec3.h"
#include "material.h"

class MTLParser {
    public:

    /**
     * @brief parses through MTL file, creating a vector material objects.
     * Also has a map of keys allowing you to find the index in the vector given a string.
    */
    bool parse(std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            return false;
        }

        int i = 0;

        std::shared_ptr<material> currentMaterial = nullptr;

        std::string line;
        while (getline(file, line)) {
            std::istringstream lineStream(line);
            std::string lineType;
            lineStream >> lineType;

            if (lineType == "newmtl") {

                if (currentMaterial != nullptr) {
                    // Save the previous material before starting a new one
                    materials.push_back(currentMaterial);
                }

                std::string mat_id;
                lineStream >> mat_id;

                matKeys[mat_id] = i;
                i++;
            } else if (lineType == "Kd") {
                float r, g, b;
                lineStream >> r >> g >> b;
                currentMaterial = make_shared<lambertian>(color(r,g,b));
            }
        }

        if (currentMaterial != nullptr) {
            materials.push_back(currentMaterial);
        }


        return true;
    }
    
    const std::vector<shared_ptr<material>>& getMaterials() {
        return materials;
    }

    const std::map<std::string, int>& getMatKeys() {
        return matKeys;
    }

    private:
    std::vector<shared_ptr<material>> materials;
    std::map<std::string, int> matKeys;

};

#endif