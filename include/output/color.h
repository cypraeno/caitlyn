#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

/**
 * @brief Converts a floating-point color value to a 256-scale integer suitable for image formats.
 * 
 * This function applies gamma correction by using a square root and scales the color based on the number of samples per pixel.
 * 
 * @param c The color to convert
 * @param samples_per_pixel The number of samples per pixel used in the rendering, used for average color calculation.
 * @return `color` The gamma-corrected color scaled to [0, 255] suitable for most image formats.
 */
color color_to_256(color c, int samples_per_pixel);

/**
 * @brief Outputs a color in 256-scale format to an output stream, used for ppm output.
 * 
 * This function converts the floating-point representation of a color to an integer scale and writes it to the given output stream.
 * 
 * @param out The output stream to write the color to.
 * @param pixel_color The color of a pixel, in floating-point format.
 * @param samples_per_pixel The number of samples per pixel, which affects color scaling.
 */
void write_color(std::ostream &out, color pixel_color, int samples_per_pixel);

#endif