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
    noise_texture();

    noise_texture(double sc);

    color value(double u, double v, const point3& p) const override;

  private:
    perlin noise;
    double scale;
};


class solid_color : public texture {
  public:
    solid_color(color c);
    solid_color(double red, double green, double blue);
   
    color value(double u, double v, const point3& p) const override;

  private:
    color color_value;
};

class checker_texture : public texture {
  public:
    checker_texture(double _scale, shared_ptr<texture> _even, shared_ptr<texture> _odd);
    checker_texture(double _scale, color c1, color c2);

    color value(double u, double v, const point3& p) const override;

  private:
    double inv_scale;
    shared_ptr<texture> even;
    shared_ptr<texture> odd;
};

class image_texture : public texture {
  public:
    image_texture(const char* filename);

    color value(double u, double v, const point3& p) const;

  private:
    rtw_image image;
};

#endif
