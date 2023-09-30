#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

using std::sqrt;


/**
 * implemntation of a 3D vector
 */
class vec3 {

    // constructors
    public:
        __host__ __device__ vec3() : e{0,0,0} {}                                            // initialize 3D vector at origin

        __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}        // initialize 3D vector with custom coordinates

        __host__ __device__ float x() const { return e[0]; }
        __host__ __device__ float y() const { return e[1]; }
        __host__ __device__ float z() const { return e[2]; }

    // operator overloading
    public:

        __host__ __device__ inline const vec3& operator+() const { return *this; }                 // addition
        __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }    // inverse
        __host__ __device__ inline float operator[](int i) const { return e[i]; }                  // const index
        __host__ __device__ inline float& operator[](int i) { return e[i]; };                      // mutatable index

        __host__ __device__ inline vec3& operator+=(const vec3 &v2);                               // self addition
        __host__ __device__ inline vec3& operator-=(const vec3 &v2);                               // self subtraction
        __host__ __device__ inline vec3& operator*=(const vec3 &v2);                               // self vector divide
        __host__ __device__ inline vec3& operator/=(const vec3 &v2);                               // self vector multiply
        __host__ __device__ inline vec3& operator*=(const float t);                                // self scalar multiplcation
        __host__ __device__ inline vec3& operator/=(const float t);                                // self scalar division

    // methods
    public:
        /// return the length of a 3D vector.
        __host__ __device__ float length() const {
            return sqrt(length_squared());
        }

    // attributes
    public:
        float e[3];
};


// vec3 utility functions

/**
 * receive 3D vector coordinates as input
 * 
 * @param[in] in stream to receive input from
 * @param[out] v vector to write to
 * 
 * @return inputted corrdinates of v
 * 
 * @relatesalso vec3
*/
inline std::istream& operator>>(std::istream &is, vec3 &v) {
    is >> v.e[0] >> v.e[1] >> v.e[2];
    return is;
}

/**
 * output 3D vector coordinates
 * 
 * @param[in] out stream to output to
 * @param[in] v vector to output 
 * 
 * @return coordinates of v
 * 
 * @relatesalso vec3
*/
inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {       
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {       
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}
__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v){
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v){
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v){
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
    float k = 1.0/t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}


// type aliases for vec3
using point3 = vec3;    // 3D point
using colour = vec3;    // RGB color

#endif
