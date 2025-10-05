#ifndef __PSTD_TRIANGLE_HPP__
#define __PSTD_TRIANGLE_HPP__

#include "vec3.h"
#include "hitable.h"
#include <cuda_runtime.h>
#include <math_functions.h>


class triangle : public hitable {
public:
	__device__ triangle() {}
	__device__ triangle(vec3 v0, vec3 v1, vec3 v2, material* m) {
		vertex[0] = v0;
		vertex[1] = v1;
		vertex[2] = v2;

		e1 = v1 - v0;
		e2 = v2 - v0;

		normal = unit_vector(cross(e1, e2));
		mat = m;
	}

	__device__ virtual bool hit(const ray& r, float tmin, float tmax, surface_interaction& si) const;

	vec3 vertex[3];
	vec3 e1, e2;
	vec3 normal;
	material* mat;
};

__device__ bool triangle::hit(const ray& r, float tmin, float tmax, surface_interaction& si) const {
	vec3 tvec = r.origin() - vertex[0];
	vec3 pvec = cross(r.direction(), e2);
	float det = dot(e1, pvec);
	det = __fdividef(1.0f, det);

	float u = dot(tvec, pvec) * det;
	if (u < 0.0f || u > 1.0f) return false;

	vec3 qvec = cross(tvec, e1);
	float v = dot(r.direction(), qvec) * det;
	if (v < 0.0f || u + v > 1.0f) return false;

	si.t = dot(e2, qvec) * det;
	return true;
}

#endif