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
#include "light.h"
#include "texture.h"
#include "sphere_primitive.h"
#include "quad_primitive.h"
#include "scene.h"

/**
 * @class CSRParser
 * @brief A parser that constructs Scene objects by reading a given CSR (Caitlyn Scene Representation) file.
 *
 * Call parseCSR(parseCSR(const std::string& filePath, RTCDevice device) to get a std::shared_ptr<Scene>
 * parseCSR does NOT validate the formatting and structure of the CSR file, and may result in a seg fault.
*/
class CSRParser {
public:
    std::ifstream file;

    /**
     * @brief Parses a CSR file and returns a std::shared_ptr<Scene> WITHOUT the scene committed.
     * The user must call scene_ptr->commitScene(); and rtcReleaseDevice(device);.
    */
    std::shared_ptr<Scene> parseCSR(const std::string& filePath, RTCDevice device) {
        file = std::ifstream(filePath);
        std::string line;
        std::map<std::string, std::shared_ptr<material>> materials;
        std::map<std::string, std::shared_ptr<texture>> textures;
        
        if (!file.is_open() || !file.good()) {
            rtcReleaseDevice(device);
            throw std::runtime_error("Could not open file: " + filePath);
        }

        // Read in version
        getNextLine(file, line);
        if (trim(line) != "version 0.1.2") {
            rtcReleaseDevice(device);
            throw std::runtime_error("Unsupported version or missing version marker");
        }

        Camera cam = readCamera();
        auto scene_ptr = make_shared<Scene>(device, cam);

        while (getNextLine(file, line)) {
            line = trim(line);
            if (startsWith(line, "Material")) {
                // Extract material ID from brackets (e.g., Material[Lambertian] -> Lambertian)
                auto idStart = line.find('[') + 1;
                auto idEnd = line.find(']');
                std::string materialType = line.substr(idStart, idEnd - idStart);
                if (materialType == "Lambertian") {
                    std::string materialId, texture;
                    getNextLine(file, materialId); getNextLine(file, texture);
                    std::string texture_id = readStringProperty(texture);
                    if (texture_id == "no") {
                        std::string albedo;
                        getNextLine(file, albedo);
                        materials[readStringProperty(materialId)] = std::make_shared<lambertian>(readXYZProperty(albedo));  
                    } else {
                        materials[readStringProperty(materialId)] = std::make_shared<lambertian>(textures[texture_id]);
                    }
                } else if (materialType == "Metal") {
                    std::string materialId, albedo, fuzz;
                    getNextLine(file, materialId); getNextLine(file, albedo); getNextLine(file, fuzz);
                    materials[readStringProperty(materialId)] = std::make_shared<metal>(readXYZProperty(albedo), readDoubleProperty(fuzz));
                } else if (materialType == "Dielectric") {
                    std::string materialId, ir;
                    getNextLine(file, materialId); getNextLine(file, ir);
                    materials[readStringProperty(materialId)] = std::make_shared<dielectric>(readDoubleProperty(ir));
                } else if (materialType == "Emissive") {
                    std::string materialId, rgb, strength;
                    getNextLine(file, materialId); getNextLine(file, rgb); getNextLine(file, strength);
                    materials[readStringProperty(materialId)] = std::make_shared<emissive>( (readDoubleProperty(strength) * readXYZProperty(rgb)) );
                } else {
                    rtcReleaseDevice(device);
                    throw std::runtime_error("Material type UNDEFINED: Material[Lambertian|Metal|Dielectric|Emissive]");
                }
            } else if (startsWith(line, "Texture")) {
                auto idStart = line.find('[') + 1;
                auto idEnd = line.find(']');
                std::string textureType = line.substr(idStart, idEnd - idStart);
                if (textureType == "Checker") {
                    std::string textureId, scale, c1, c2;
                    getNextLine(file, textureId); getNextLine(file, scale); getNextLine(file, c1); getNextLine(file, c2);
                    textures[readStringProperty(textureId)] = std::make_shared<checker_texture>(readDoubleProperty(scale), readXYZProperty(c1), readXYZProperty(c2));
                } else if (textureType == "Image") {
                    std::string textureId, path;
                    getNextLine(file, textureId); getNextLine(file, path);
                    textures[readStringProperty(textureId)] = std::make_shared<image_texture>(readStringProperty(path).c_str());
                } else if (textureType == "Noise") {
                    std::string textureId, scale;
                    getNextLine(file, textureId); getNextLine(file, scale);
                    textures[readStringProperty(textureId)] = std::make_shared<noise_texture>(readDoubleProperty(scale));
                } else {
                    rtcReleaseDevice(device);
                    throw std::runtime_error("Texture type UNDEFINED: Texture[Checker|Image|Noise]");
                }
            } else if (startsWith(line, "Sphere")) {
                std::string position, material, radius;
                getNextLine(file, position); getNextLine(file, material); getNextLine(file, radius);
                auto sphere = make_shared<SpherePrimitive>(readXYZProperty(position), materials[readStringProperty(material)], readDoubleProperty(radius), device);
                scene_ptr->add_primitive(sphere);
            } else if (startsWith(line, "Quad")) {
                std::string position, u, v, material;
                getNextLine(file, position); getNextLine(file, u); getNextLine(file, v); getNextLine(file, material);
                auto quad = make_shared<QuadPrimitive>(readXYZProperty(position), readXYZProperty(u), readXYZProperty(v), materials[readStringProperty(material)], device);
                scene_ptr->add_primitive(quad);
            }
        }

        return scene_ptr;
    }
private:
    /**
     * @brief Given a string, reads in the next line from the file while accounting for comments,
     * If the line begins with a '#', it is a comment and skips to the next line.
     * String is also preprocessed to ignore anything after a #.
    */
    bool getNextLine(std::ifstream& file, std::string& holder) {
        if (!getline(file, holder)) {return false;};
        while (startsWith(holder, "#")) { // if line still is a comment
            if (!getline(file, holder)) {return false;};
        }
        std::vector<std::string> tokens = split(holder, '#');
        holder = trim(tokens[0]);
        return true;
    }

    // Helper String Functions
    std::vector<std::string> split(const std::string &s, char delimiter = ' ') {
        std::vector<std::string> tokens;

        if (s.empty()) {
            // Return a vector with a single, empty string
            tokens.push_back("");
            return tokens;
        }
        std::istringstream tokenStream(s);
        std::string token;
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }

        if (tokens.empty()) {
            tokens.push_back(s);
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
        while (getNextLine(file, line)) {
            if (startsWith(line, "Camera")) {
                getNextLine(file, lookfrom); getNextLine(file, lookat); getNextLine(file, vup);
                getNextLine(file, vfov); getNextLine(file, aspect_ratio); getNextLine(file, aperture); getNextLine(file, focus_dist);
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
