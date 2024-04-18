#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

color color_to_256(color c, int samples_per_pixel) {
    auto r = c.x();
    auto g = c.y();
    auto b = c.z();
    
    auto scale = 1.0 / samples_per_pixel;

    r = 256 * clamp(sqrt(scale * r), 0.0, 0.999);
    g = 256 * clamp(sqrt(scale * g), 0.0, 0.999);
    b = 256 * clamp(sqrt(scale * b), 0.0, 0.999);

    return color(r, g, b);
}

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {
    color to_256 = color_to_256(pixel_color, samples_per_pixel);

    out << static_cast<int>(to_256.x()) << ' '
        << static_cast<int>(to_256.y()) << ' '
        << static_cast<int>(to_256.z()) << '\n';
}

#endif