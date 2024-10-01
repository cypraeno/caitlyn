#define STB_IMAGE_IMPLEMENTATION
#include "image.hh"

image::image() : data(nullptr) {}

image::image(const char* image_filename, int bytes_per_pixel) : bytes_per_pixel{bytes_per_pixel} {
    auto filename = std::string(image_filename);
    auto imagedir = getenv("IMAGES");

    // Hunt for the image file in some likely locations.
    if (imagedir && load(std::string(imagedir) + "/" + image_filename)) return;
    if (load(filename)) return;

    std::cerr << "ERROR: Could not load image file '" << image_filename << "'.\n";
}

image::~image() {
    STBI_FREE(data);
}

bool image::load(const std::string filename) {
    auto n = bytes_per_pixel; // Dummy out parameter: original components per pixel
    data = stbi_load(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
    bytes_per_scanline = image_width * bytes_per_pixel;
    return data != nullptr;
}

int image::width()  const { return (data == nullptr) ? 0 : image_width; }
int image::height() const { return (data == nullptr) ? 0 : image_height; }

const unsigned char* image::pixel_data(int x, int y) const {
    static unsigned char magenta[] = { 255, 0, 255 };
    if (data == nullptr) return magenta;

    x = clamp(x, 0, image_width);
    y = clamp(y, 0, image_height);

    return data + y*bytes_per_scanline + x*bytes_per_pixel;
}

int image::clamp(int x, int low, int high) {
    // Return the value clamped to the range [low, high).
    if (x < low) return low;
    if (x < high) return x;
    return high - 1;
}
