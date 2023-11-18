#include "eRay.h"

unsigned int Ray::id = 0; 

Ray::Ray(const point3& org, const vec3& dir, float time, float tnear, float tfar, unsigned int mask, unsigned int flag) :
    org{org}, dir{dir}, time{time}, tnear{tnear}, tfar{tfar}, mask{mask}, flag{flag} { 
    
    ++this->id; 
}

point3 Ray::getOrg() const { return this->org; }
vec3 Ray::getDir() const { return this->dir; }
float Ray::getTime() const { return this->time; }
float Ray::getTNear() const { return this->tnear; }
float Ray::getTFar() const { return this->tfar; }
unsigned int Ray::getID() const { return this->id; }
unsigned int Ray::getFlags() const {return this->flags; }

point3 Ray::at(float t) const { return this->org + t*this->dir; }

void Ray::createRTCRay(struct RTCRay& ray) const {

    ray.org_x = this->org.x();
    ray.org_y = this->org.y();
    ray.org_z = this->org.z();
    ray.tnear = this->tnear;

    ray.dir_x = this->dir.x();
    ray.dir_y = this->dir.y();
    ray.dir_z = this->dir.z();
    ray.time = this->time;

    ray.tfar = this->tfar;
    ray.mask = this->mask;
    ray.id = this->id;
    ray.flags = this->flags;
}
