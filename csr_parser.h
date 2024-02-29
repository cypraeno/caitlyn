#ifndef CSRPARSER_H
#define CSRPARSER_H

#include <embree4/rtcore.h>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <optional>
#include "camera.h"
#include "material.h"
#include "sphere_primitive.h"
#include "quad_primitive.h"
#include "scene.h"

class CSRParser {
public:
    std::ifstream file;

    std::shared_ptr<Scene> parseCSR(const std::string& filePath, RTCDevice device) {
        file = std::ifstream(filePath);
        std::string line;
        std::map<std::string, std::shared_ptr<material>> materials;
        
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filePath);
        }

        // Read in versionn
        getline(file, line);
        if (trim(line) != "version 0.1.0") {
            throw std::runtime_error("Unsupported version or missing version marker");
        }

        Camera cam = readCamera();
        auto scene_ptr = make_shared<Scene>(device, cam);

        while (getline(file, line)) {
            line = trim(line);
            if (startsWith(line, "Material")) {
                // Extract material ID from brackets (e.g., Material[Lambertian] -> Lambertian)
                auto idStart = line.find('[') + 1;
                auto idEnd = line.find(']');
                std::string materialType = line.substr(idStart, idEnd - idStart);
                if (materialType == "Lambertian") {
                    std::string materialId, texture, albedo;
                    getline(file, materialId); getline(file, texture); getline(file, albedo);
                    if (readStringProperty(texture) == "no") {
                        materials[readStringProperty(materialId)] = std::make_shared<lambertian>(readXYZProperty(albedo));  
                    }  
                }
            } else if (startsWith(line, "Sphere")) {
                std::string position, material, radius;
                getline(file, position); getline(file, material); getline(file, radius);
                auto sphere = make_shared<SpherePrimitive>(readXYZProperty(position), materials[readStringProperty(material)], readDoubleProperty(radius), device);
                scene_ptr->add_primitive(sphere);
            } else if (startsWith(line, "Quad")) {
                std::string position, u, v, material;
                getline(file, position); getline(file, u); getline(file, v); getline(file, material);
                auto quad = make_shared<QuadPrimitive>(readXYZProperty(position), readXYZProperty(u), readXYZProperty(v), materials[readStringProperty(material)], device);
                scene_ptr->add_primitive(quad);
            }
        }

        return scene_ptr;
    }
private:
    // Helper String Functions
    std::vector<std::string> split(const std::string &s, char delimiter = ' ') {
        std::vector<std::string> tokens;
        std::istringstream tokenStream(s);
        std::string token;
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }


    std::string trim(const std::string& str) {
        // Include '\r' and '\n' in the character set for trimming
        size_t first = str.find_first_not_of(" \r\n");
        if (std::string::npos == first) {
            // Return an empty string if only whitespace characters are found
            return "";
        }
        size_t last = str.find_last_not_of(" \r\n");
        return str.substr(first, (last - first + 1));
    }

    bool startsWith(const std::string& str, const std::string& prefix) {
        return str.size() >= prefix.size() &&
            str.compare(0, prefix.size(), prefix) == 0;
    }

    point3 readXYZProperty(std::string line) {
        auto tokens = split(line);
        point3 p(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        return p;
    }

    double readDoubleProperty(std::string line) {
        auto tokens = split(line);
        return std::stod(tokens[1]);
    }

    std::string readStringProperty(std::string line) {
        auto tokens = split(line);
        return tokens[1];
    }

    double readRatioProperty(std::string line) {
        auto tokens = split(line);
        auto ratio_tokens = split(tokens[1], '/');
        return std::stod(ratio_tokens[0]) / std::stod(ratio_tokens[1]);
    }

    // Reading in objects
    Camera readCamera() {
        std::string line;
        std::string lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist;
        while (getline(file, line)) {
            if (startsWith(line, "Camera")) {
                getline(file, lookfrom); getline(file, lookat); getline(file, vup);
                getline(file, vfov); getline(file, aspect_ratio); getline(file, aperture); getline(file, focus_dist);
                break;
            }
        }
        Camera cam(readXYZProperty(lookfrom), readXYZProperty(lookat),
            readXYZProperty(vup), readDoubleProperty(vfov),
            readRatioProperty(aspect_ratio), readDoubleProperty(aperture),
            readDoubleProperty(focus_dist));
        return cam;
    }

    
};

#endif
