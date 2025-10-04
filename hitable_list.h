#ifndef __PSTD__HITABLE_LIST_HPP__
#define __PSTD__HITABLE_LIST_HPP__

#include "hitable.h"

class hitable_list : public hitable {
public:
	__device__ hitable_list() {}
	__device__ hitable_list(hitable** l, int n) { list = l; list_size = n; }
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, surface_interaction& si) const;
	hitable** list;
	int list_size;
};

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, surface_interaction& si) const {
	surface_interaction temp_si;
	bool hit_anything = false;
	float closest_so_far = t_max;
	for (int i = 0; i < list_size; i++) {
		if (list[i]->hit(r, t_min, closest_so_far, temp_si)) {
			hit_anything = true;
			closest_so_far = temp_si.t;
			si = temp_si;
		}
	}
	return hit_anything;
}

#endif