#ifndef MATERIAL_H
#define MATERIAL_H

#include "general.h"

struct hit_record;

class material {
    public:
        virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

#endif