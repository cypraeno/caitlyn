#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"


/**
 * implementation of a list of hittable objects
 * 
 * the list inherits from the hittable class and is used to create goups 
 * of objects which can be hit in a scene
*/
class hitable_list: public hitable  {

    // constructors
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable **l, int n) {list = l; list_size = n; }
    
    //methods
    public:
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;   // anything hit?
    
    // attributes
    public:
        hitable **list;
        int list_size;
};


/**
 * determines if an object in the list was hit and records the hit if it was
 * 
 * @param[in] r the ray being shot out from the eye
 * @param[in] t_min, t_max the interval of acceptable intersections between the ray and objects
 * @param[out] rec the details of the intersection
 * 
 * @return if any objects were hit
 * 
 * @relatesalso hittable_list
*/
__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
}

#endif
