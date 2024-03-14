#include <png.h>
#include <vector>
#include <cstdio>
#include "color.h"

uint8_t to_byte(float value, int samples_per_pixel) {
    auto scale = 1.0 / samples_per_pixel;
    value = sqrt(scale * value);
    return static_cast<uint8_t>(255.99 * clamp(value, 0.0, 0.999));
}

void write_png(const char* filename, int width, int height, int samples_per_pixel, const std::vector<color>& buffer) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) return;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) return;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        fclose(fp);
        return;
    }

    // Error handling
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);

    // Write image data
    png_bytep row = (png_bytep) malloc(3 * width * sizeof(png_byte));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            color col = buffer[y * width + x];
            row[x*3 + 0] = to_byte(col.x(), samples_per_pixel);
            row[x*3 + 1] = to_byte(col.y(), samples_per_pixel);
            row[x*3 + 2] = to_byte(col.z(), samples_per_pixel);
        }
        png_write_row(png_ptr, row);
    }
    free(row);

    // End write
    png_write_end(png_ptr, nullptr);

    // Clean up
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
