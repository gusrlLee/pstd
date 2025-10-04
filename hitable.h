#ifndef __PSTD_HITABLE_HPP__
#define __PSTD_HITABLE_HPP__

#include "ray.h"

struct surface_interaction {
	float t;
	vec3 p;
	vec3 n;
};

class hitable {
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, surface_interaction& si) const = 0;
};

#endif