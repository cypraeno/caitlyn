#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

#include "general.h"

using std::sqrt;
using std::fabs;

/** @brief implementation of a 3D vector class */
class vec3 {

    float e[3] = {0, 0, 0};                 /**< a size-3 float array containing the vector coords in [x, y, z] */

    public:

        vec3();                             /**< default constructor */
        vec3(float x, float y, float z);    

        float x() const;                    /**< @returns vec3 x coord */
        float y() const;                    /**< @returns vec3 y coord */
        float z() const;                    /**< @returns vec3 z coord */

        // indexing overloads
        float operator[](int i) const;
        float& operator[](int i);

        // member arithmetic overloads
        vec3 operator-() const;
        vec3& operator+=(const vec3 &v);
        vec3& operator*=(const float t);
        vec3& operator/=(const float t);

        /** @return the vec3 length */
        float length() const;
        /** @return the vec3 length squared */
        float length_squared() const;
        /** @return the vec3 unit vector */
        vec3 unit_vector() const;

        /** @return a random vec3 object */
        static vec3 random();

        /**
         * @param[in] min,max the interval that all vec3 fields will be generated between
         * 
         * @return a random vec3 object within the range of min, max 
         */
        static vec3 random(float min, float max);

        /** @return a random unit vector */
        static vec3 random_unit();

        /** @return if the vec3 object is near zero */
        bool near_zero() const;
};

using point3 = vec3;   /**< @brief alias of vec3 for a 3D Point */
using color = vec3;    /**< @brief alias of vec3 for RGB Colour */

// non-member arithmetic overloads
vec3 operator+(const vec3 &u, const vec3 &v);
vec3 operator-(const vec3 &u, const vec3 &v);
vec3 operator*(const vec3 &u, const vec3 &v);
vec3 operator*(float t, const vec3 &v);
vec3 operator*(const vec3 &v, float t);
vec3 operator/(vec3 v, float t);

// vector multiplication
float dot(const vec3 &u, const vec3 &v);
vec3 cross(const vec3 &u, const vec3 &v);

/** @brief overloads std::ostream& operator<< to support vec3s */
std::ostream& operator<<(std::ostream &out, const vec3 &v);

// random vec3 generation
vec3 random_unit_vector();
vec3 random_in_unit_sphere();
vec3 random_in_hemisphere(const vec3& normal);
vec3 random_in_unit_disk();

// reflection and refraction
vec3 reflect(const vec3& v, const vec3& n);
vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat);

#endif
