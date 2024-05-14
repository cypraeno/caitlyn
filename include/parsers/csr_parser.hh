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
#include "box_primitive.h"
#include "scene.h"
#include "instances.h"

// From csr-schema lib
#include "csr_validator.hh"

/**
 * @class CSRParser
 * @brief A parser that constructs Scene objects by reading a given CSR (Caitlyn Scene Representation) file.
 *
 * Call parseCSR(parseCSR(const std::string& filePath, RTCDevice device) to get a std::shared_ptr<Scene>
 * parseCSR does NOT validate the formatting and structure of the CSR file, and may result in a seg fault without prior validation.
*/
class CSRParser {

    public:

        std::ifstream file;

        /**
         * @brief Parses a CSR file and returns a std::shared_ptr<Scene> WITHOUT the scene committed.
         * The user must call scene_ptr->commitScene(); and rtcReleaseDevice(device);.
        */
        std::shared_ptr<Scene> parseCSR(std::string& filePath, RTCDevice device);

    private:

        /**
         * @brief Given a string, reads in the next line from the file while accounting for comments,
         * If the line begins with a '#', it is a comment and skips to the next line.
         * String is also preprocessed to ignore anything after a #.
        */
        bool getNextLine(std::ifstream& file, std::string& holder);

        // Helper String Functions
        std::vector<std::string> split(const std::string &s, char delimiter = ' ');
        std::string trim(const std::string& str);
        bool startsWith(const std::string& str, const std::string& prefix);

        // Property Readers
        point3 readXYZProperty(std::string line);
        double readDoubleProperty(std::string line);
        std::string readStringProperty(std::string line);
        double readRatioProperty(std::string line);
        Camera readCamera();
};

#endif
