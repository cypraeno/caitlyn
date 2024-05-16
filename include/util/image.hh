#ifndef IMAGE_H
#define IMAGE_H

// Disable strict warnings for this header from the Microsoft Visual C++ compiler.
#ifdef _MSC_VER
    #pragma warning (push, 0)
#endif

#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#include <cstdlib>
#include <iostream>

class image {

  public:

    image();

    // Loads image data from the specified file. If the IMAGES environment variable is
    // defined, looks only in that directory for the image file. If the image was not found,
    // searches for the specified image file first from the current directory, then in the
    // images/ subdirectory, then the _parent's_ images/ subdirectory, and then _that_
    // parent, on so on, for six levels up. If the image was not loaded successfully,
    // width() and height() will return 0.
    image(const char* image_filename, int bytes_per_pixel = 3);

    ~image();

    // Loads image data from the given file name. Returns true if the load succeeded.
    bool load(const std::string filename);

    int width()  const;
    int height() const;

    // Return the address of the three bytes of the pixel at x,y (or magenta if no data).
    const unsigned char* pixel_data(int x, int y) const;

  private:
    int bytes_per_pixel = 3;
    unsigned char *data;
    int image_width, image_height;
    int bytes_per_scanline;

    // Return the value clamped to the range [low, high).
    static int clamp(int x, int low, int high);
};

// Restore MSVC compiler warnings
#ifdef _MSC_VER
    #pragma warning (pop)
#endif

#endif
