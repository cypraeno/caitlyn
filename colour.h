#ifndef COLOUR_H
#define COLOUR_H

#include "vec3.h"

#include <iostream>


/// clamps a value to be between a given interval
inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

/**
 * output a single pixel's colour
 * 
 * @param[in] out stream to output to
 * @param[in] pixel_colour colour of pixel to output
 */
void write_colour(std::ostream &out, colour pixel_colour, int pixel_samples) {

    // divide colour by number of samples
    float scale = 1.0 / pixel_samples;
    float r = pixel_colour.x() * pixel_samples;
    float g = pixel_colour.y() * pixel_samples;
    float b = pixel_colour.z() * pixel_samples;

    // write the translated [0,255] value of each colour component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

#endif
