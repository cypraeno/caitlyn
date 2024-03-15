#ifndef TEXTURE_H
#define TEXTURE_H

#include "general.h"
#include "rtw_stb_image.h"
#include "perlin.h"


class texture {
  public:
    virtual ~texture() = default;

    virtual color value(double u, double v, const point3& p) const = 0;
};

class noise_texture : public texture {
  public:
    noise_texture() {}

    noise_texture(double sc) : scale(sc) {}

    color value(double u, double v, const point3& p) const override {
        auto s = scale * p;
        return color(1,1,1) * 0.5 * (1 + sin(s.z() + 10*noise.turb(s)));

      }

  private:
    perlin noise;
    double scale;
};


class solid_color : public texture {
  public:
    solid_color(color c) : color_value(c) {}
    solid_color(double red, double green, double blue) : solid_color(color(red,green,blue)) {}
   
    color value(double u, double v, const point3& p) const override {
        return color_value;
    }

  private:
    color color_value;
};

class checker_texture : public texture {
  public:
    checker_texture(double _scale, shared_ptr<texture> _even, shared_ptr<texture> _odd)
      : inv_scale(1.0 / _scale), even(_even), odd(_odd) {}

    checker_texture(double _scale, color c1, color c2)
      : inv_scale(1.0 / (_scale/2)),
        even(make_shared<solid_color>(c1)),
        odd(make_shared<solid_color>(c2))
    {}

    color value(double u, double v, const point3& p) const override {
        auto xInteger = static_cast<int>(std::floor(inv_scale * p.x()));
        auto yInteger = static_cast<int>(std::floor(inv_scale * p.y()));
        auto zInteger = static_cast<int>(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

  private:
    double inv_scale;
    shared_ptr<texture> even;
    shared_ptr<texture> odd;
};

class image_texture : public texture {
  public:
    image_texture(const char* filename) : image(filename) {}

    color value(double u, double v, const point3& p) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (image.height() <= 0) return color(0,1,1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        if(u < 0) u = 0;
        else if(u > 1) u = 1;

        if(v < 0) v = 0;
        else if(v > 1) v = 1;
        else v = 1 - v;
        auto i = static_cast<int>(u * image.width());
        auto j = static_cast<int>(v * image.height());
        auto pixel = image.pixel_data(i,j);

        auto color_scale = 1.0 / 255.0;
        return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
    }

  private:
    rtw_image image;
};

#endif
