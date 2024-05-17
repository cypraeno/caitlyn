#include "csr_parser.hh"

std::shared_ptr<Scene> CSRParser::parseCSR(std::string& filePath, RTCDevice device)  {

    isCSR(filePath);

    file = std::ifstream(filePath);
    std::string line;
    std::map<std::string, std::shared_ptr<material>> materials;
    std::map<std::string, std::shared_ptr<texture>> textures;
    std::map<std::string, std::shared_ptr<Primitive>> primitives;
    
    if (!file.is_open() || !file.good()) {
        rtcReleaseDevice(device);
        throw std::runtime_error("Could not open file: " + filePath);
    }

    // Read in version
    getNextLine(file, line);
    if (trim(line) != "version 0.1.5") {
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
                    std::shared_ptr<PixelImageTexture> plamb = std::dynamic_pointer_cast<PixelImageTexture>(textures[texture_id]);
                    if (plamb) { // texture is a pixel lambert
                        materials[readStringProperty(materialId)] = std::make_shared<pixel_lambertian>(plamb);
                    } else { // texture is a normal lambert
                        materials[readStringProperty(materialId)] = std::make_shared<lambertian>(textures[texture_id]);
                    }
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
                std::string textureId, transparency, path;
                getNextLine(file, textureId); getNextLine(file, transparency); getNextLine(file, path);
                if (readBooleanProperty(transparency)) textures[readStringProperty(textureId)] = std::make_shared<PixelImageTexture>(readStringProperty(path).c_str());
                else textures[readStringProperty(textureId)] = std::make_shared<image_texture>(readStringProperty(path).c_str());
            } else if (textureType == "Noise") {
                std::string textureId, scale;
                getNextLine(file, textureId); getNextLine(file, scale);
                textures[readStringProperty(textureId)] = std::make_shared<noise_texture>(readDoubleProperty(scale));
            } else {
                rtcReleaseDevice(device);
                throw std::runtime_error("Texture type UNDEFINED: Texture[Checker|Image|Noise]");
            }
        } else if (startsWith(line, "Sphere")) {
            std::string id, position, material, radius;
            getNextLine(file, id); getNextLine(file, position); getNextLine(file, material); getNextLine(file, radius);
            auto sphere = make_shared<SpherePrimitive>(readXYZProperty(position), materials[readStringProperty(material)], readDoubleProperty(radius), device);
            primitives[readStringProperty(id)] = sphere;
            scene_ptr->add_primitive(sphere);
        } else if (startsWith(line, "Quad")) {
            std::string id, position, u, v, material;
            getNextLine(file, id); getNextLine(file, position); getNextLine(file, u); getNextLine(file, v); getNextLine(file, material);
            auto quad = make_shared<QuadPrimitive>(readXYZProperty(position), readXYZProperty(u), readXYZProperty(v), materials[readStringProperty(material)], device);
            primitives[readStringProperty(id)] = quad;
            scene_ptr->add_primitive(quad);
        } else if (startsWith(line, "Instance")) {
            auto idStart = line.find('[') + 1;
            auto idEnd = line.find(']');
            std::string instanceType = line.substr(idStart, idEnd - idStart);
            if (instanceType == "SpherePrimitive") {
                std::string prim_id, translate;
                getNextLine(file, prim_id); getNextLine(file, translate);
                vec3 translateVector = readXYZProperty(translate);
                float transform[12] = {
                    1, 0, 0, translateVector.x(),
                    0, 1, 0, translateVector.y(),
                    0, 0, 1, translateVector.z()
                };
                std::shared_ptr<SpherePrimitive> instance_ptr = std::dynamic_pointer_cast<SpherePrimitive>(primitives[readStringProperty(prim_id)]);
                if (!instance_ptr) {
                    rtcReleaseDevice(device);
                    throw std::runtime_error("Instance key ERROR: " + readStringProperty(prim_id) + " is not a SpherePrimitive!");
                }
                auto instance = make_shared<SpherePrimitiveInstance>(instance_ptr, transform, device);
                scene_ptr->add_primitive_instance(instance, device);
            } else if (instanceType == "QuadPrimitive") {
                std::string prim_id, translate;
                getNextLine(file, prim_id); getNextLine(file, translate);
                vec3 translateVector = readXYZProperty(translate);
                float transform[12] = {
                    1, 0, 0, translateVector.x(),
                    0, 1, 0, translateVector.y(),
                    0, 0, 1, translateVector.z()
                };
                std::shared_ptr<QuadPrimitive> instance_ptr = std::dynamic_pointer_cast<QuadPrimitive>(primitives[readStringProperty(prim_id)]);
                if (!instance_ptr) {
                    rtcReleaseDevice(device);
                    throw std::runtime_error("Instance key ERROR: " + readStringProperty(prim_id) + " is not a QuadPrimitive!");
                }
                auto instance = make_shared<QuadPrimitiveInstance>(instance_ptr, transform, device);
                scene_ptr->add_primitive_instance(instance, device);
            } else {
                rtcReleaseDevice(device);
                throw std::runtime_error("Instance type UNDEFINED: Instance[SpherePrimitive|QuadPrimitive]");
            }
        }
    }

    return scene_ptr;
}

bool CSRParser::getNextLine(std::ifstream& file, std::string& holder) {
    if (!getline(file, holder)) {return false;};
    while (startsWith(holder, "#")) { // if line still is a comment
        if (!getline(file, holder)) {return false;};
    }
    std::vector<std::string> tokens = split(holder, '#');
    holder = trim(tokens[0]);
    return true;
}

std::vector<std::string> CSRParser::split(const std::string &s, char delimiter) {
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

std::string CSRParser::trim(const std::string& str) {
    // Include '\r' and '\n' in the character set for trimming
    size_t first = str.find_first_not_of(" \r\n");
    if (std::string::npos == first) {
        // Return an empty string if only whitespace characters are found
        return "";
    }
    size_t last = str.find_last_not_of(" \r\n");
    return str.substr(first, (last - first + 1));
}

bool CSRParser::startsWith(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() &&
        str.compare(0, prefix.size(), prefix) == 0;
}

bool CSRParser::readBooleanProperty(std::string line) {
    auto tokens = split(line);
    bool value;
    std::istringstream(tokens[1]) >> std::boolalpha >> value;
    return value;
}

point3 CSRParser::readXYZProperty(std::string line) {
    auto tokens = split(line);
    point3 p(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
    return p;
}


double CSRParser::readDoubleProperty(std::string line) {
    auto tokens = split(line);
    return std::stod(tokens[1]);
}

std::string CSRParser::readStringProperty(std::string line) {
    auto tokens = split(line);
    return tokens[1];
}

double CSRParser::readRatioProperty(std::string line) {
    auto tokens = split(line);
    auto ratio_tokens = split(tokens[1], '/');
    return std::stod(ratio_tokens[0]) / std::stod(ratio_tokens[1]);
}

Camera CSRParser::readCamera() {
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