#include "vec3.h"

vec3::vec3() {}

vec3::vec3(float x, float y, float z) : e{x, y ,z} {}

float vec3::x() const { return this->e[0]; }
float vec3::y() const { return this->e[1]; }
float vec3::z() const { return this->e[2]; }

vec3 vec3::operator-() const { return vec3{-e[0], -e[1], -e[2]}; }

vec3& vec3::operator+=(const vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

vec3& vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

vec3& vec3::operator/=(const float t) { return *this *= 1/t; }

float vec3::operator[](int i) const { return e[i]; }

float& vec3::operator[](int i) { return e[i]; }

float vec3::length() const { return sqrt(this->length_squared()); }

float vec3::length_squared() const { 
    return (this->e[0]*this->e[0] +
            this->e[1]*this->e[1] +
            this->e[2]*this->e[2]);
}

vec3 vec3::unit_vector() const { return *this / this->length(); }

vec3 vec3::random() {
    return vec3{random_float(), random_float(), random_float()};
}

vec3 vec3::random(float min, float max) {
    return vec3{random_float(min, max), random_float(min, max), random_float(min, max)};
}

vec3 vec3::random_unit() { return vec3::random().unit_vector(); }

bool vec3::near_zero() const {

    const auto s = 1e-8;
    return (fabs(this->e[0]) < s) && (fabs(this->e[1]) < s) && (fabs(this->e[2]) < s);
}

vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3{u.x() + v.x(), u.y() + v.y(), u.z() + v.z()};
}

vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3{u.x() - v.x(), u.y() - v.y(), u.z() - v.z()};
}

vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3{u.x() * v.x(), u.y() * v.y(), u.z() * v.z()};
}

vec3 operator*(float t, const vec3 &v) {
    return vec3{t*v.x(), t*v.y(), t*v.z()};
}

vec3 operator*(const vec3 &v, float t) { return t * v; }

vec3 operator/(vec3 v, float t) { return (1/t) * v; }

float dot(const vec3 &u, const vec3 &v) {
    return (u.x() * v.x() +
            u.y() * v.y() +
            u.z() * v.z());
}

vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3{u.y() * v.z() - u.z() * v.y(),
                u.z() * v.x() - u.x() * v.z(),
                u.x() * v.y() - u.y() * v.x()};
}

std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

vec3 unit_vector(vec3 v) { return v / v.length(); }

vec3 random_unit_vector() { return unit_vector(random_in_unit_sphere()); }

vec3 random_in_unit_sphere() { 
    vec3 p = vec3::random(-1, 1);
    while (p.length_squared() >= 1) {
        p = vec3::random(-1, 1);
    }
    return p;
    //return vec3::random_unit();
 }

vec3 random_in_hemisphere(const vec3& normal) {

    vec3 in_unit_sphere = random_in_unit_sphere();

    if (dot(in_unit_sphere, normal) > 0.0)  return in_unit_sphere;  // In the same hemisphere as the normal
    else                                    return -in_unit_sphere;
}

vec3 random_in_unit_disk() {

    vec3 rand_vec = {random_float(), random_float(), 0};
    return rand_vec.unit_vector();
}

vec3 reflect(const vec3& v, const vec3& n) { return v - 2*n * dot(v, n); }

vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {

    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;

    return r_out_perp + r_out_parallel;
}
