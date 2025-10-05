#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

class material;

struct surface_interaction {
    float t;
    vec3 p;
    vec3 n;
    material *m;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, surface_interaction& si) const = 0;
};

#endif
