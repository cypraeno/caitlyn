#ifndef PNG_OUTPUT_H
#define PNG_OUTPUT_H

#include <png.h>
#include <vector>
#include <cstdio>
#include "color.h"

uint8_t to_byte(float value);

void write_png(const char* filename, int width, int height, int samples_per_pixel, const std::vector<color>& buffer);

#endif
