#include "texture.h"

noise_texture::noise_texture() {}
noise_texture::noise_texture(double sc) : scale(sc) {}

color noise_texture::value(double u, double v, const point3& p) const {
    auto s = scale * p;
    return color(1,1,1) * 0.5 * (1 + sin(s.z() + 10*noise.turb(s)));
}

solid_color::solid_color(color c) : color_value(c) {}
solid_color::solid_color(double red, double green, double blue) : solid_color(color(red, green, blue)) {}

color solid_color::value(double u, double v, const point3& p) const { return color_value; }



checker_texture::checker_texture(double _scale, shared_ptr<texture> _even, shared_ptr<texture> _odd)
      : inv_scale(1.0 / _scale), even(_even), odd(_odd) {}

checker_texture::checker_texture(double _scale, color c1, color c2)
      : inv_scale(1.0 / (_scale/2)),
        even(make_shared<solid_color>(c1)),
        odd(make_shared<solid_color>(c2)) {}

color checker_texture::value(double u, double v, const point3& p) const {
    auto xInteger = static_cast<int>(std::floor(inv_scale * p.x()));
    auto yInteger = static_cast<int>(std::floor(inv_scale * p.y()));
    auto zInteger = static_cast<int>(std::floor(inv_scale * p.z()));

    bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

    return isEven ? even->value(u, v, p) : odd->value(u, v, p);
}


image_texture::image_texture(const char* filename) : image_data(filename) {}

color image_texture::value(double u, double v, const point3& p) const {
    // If we have no texture data, then return solid cyan as a debugging aid.
    if (image_data.height() <= 0) return color(0,1,1);

    // Clamp input texture coordinates to [0,1] x [1,0]
    u = clamp(u, 0.0, 1.0);

    if (0 <= v && v <= 1) v = 1 - v;
    else v = clamp(v, 0.0, 1.0);
    
    auto i = static_cast<int>(u * image_data.width());
    auto j = static_cast<int>(v * image_data.height());
    auto pixel = image_data.pixel_data(i,j);

    auto color_scale = 1.0 / 255.0;
    return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
}

// Pixel Image Textures
PixelImageTexture::PixelImageTexture(const char* filename) : img(filename, 4) {}

color PixelImageTexture::value(double u, double v, const point3& p) const {
    throw std::runtime_error("Incorrect value func. called. Use { color4 value(double u, double v) } instead.");
    return color(0,1,1);
}

color4 PixelImageTexture::value(double u, double v) const {

    if (img.height() <= 0) throw std::invalid_argument("Image height is less than 0");

    u = clamp(u, 0.0, 1.0);

    if (0 <= v && v <= 1) v = 1 - v;
    else v = clamp(v, 0.0, 1.0);

    auto i = static_cast<int>(u * img.width());
    auto j = static_cast<int>(v * img.height());

    auto pixel = img.pixel_data(i,j);
    auto color_scale = 1.0 / 255.0;

    return color4 {
        static_cast<float>(color_scale)*pixel[3],
        color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]), 
    };
}