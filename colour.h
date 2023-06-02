#ifndef COLOUR_H
#define COLOUR_H

#include "vec3.h"

#include <iostream>


/**
 * output a single pixel's colour
 * 
 * @param[in] out stream to output to
 * @param[in] pixel_colour colour of pixel to output
 */
void write_colour(std::ostream &out, colour pixel_colour) {
    // write the translated [0,255] value of each colour component.
    out << static_cast<int>(255.999 * pixel_colour.x()) << ' '
        << static_cast<int>(255.999 * pixel_colour.y()) << ' '
        << static_cast<int>(255.999 * pixel_colour.z()) << '\n';
}

#endif
