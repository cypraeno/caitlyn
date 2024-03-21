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
#include "light.h"


/**
 * @brief provides a parse() that loads a file and populates a vector of materials.
 * Does NOT support image textures or any material parameters other than Ke and Kd.
 * Will load either lambertian or emissive_lambertian.
*/
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
        color albedo;
        bool creatingEmissive = false;
        color emission;

        std::string line;
        while (getline(file, line)) {
            std::istringstream lineStream(line);
            std::string lineType;
            lineStream >> lineType;

            if (lineType == "newmtl") {
                
                if (i != 0) { // first material
                    // Save the previous material before starting a new one
                    if (creatingEmissive) {
                        currentMaterial = make_shared<emissive_lambertian>(albedo, emission);
                    } else {
                        currentMaterial = make_shared<lambertian>(albedo);
                    }
                    creatingEmissive = false;
                    materials.push_back(currentMaterial);
                }

                std::string mat_id;
                lineStream >> mat_id;

                matKeys[mat_id] = i;
                i++;
            } else if (lineType == "Kd") {
                float r, g, b;
                lineStream >> r >> g >> b;
                albedo = color(r,g,b);
            } else if (lineType == "Ke") {
                float r, g, b;
                lineStream >> r >> g >> b;
                emission = color(r,g,b);
                creatingEmissive = true;
            }
        }

        if (creatingEmissive) {
            currentMaterial = make_shared<emissive_lambertian>(albedo, emission);
        } else {
            currentMaterial = make_shared<lambertian>(albedo);
        }
        creatingEmissive = false;
        materials.push_back(currentMaterial);


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