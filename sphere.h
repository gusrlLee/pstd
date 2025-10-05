#ifndef __PSTD_SPHERE_HPP__
#define __PSTD_SPHERE_HPP__

#include "hitable.h"

class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat(m) {}
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, surface_interaction& si) const;

	vec3 center;
	float radius;
	material* mat;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, surface_interaction& si) const {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			si.t = temp;
			si.p = r.at(si.t);
			si.n = (si.p - center) / radius;
			si.m = mat;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			si.t = temp;
			si.p = r.at(si.t);
			si.n = (si.p - center) / radius;
			si.m = mat;
			return true;
		}
	}
	return false;
}


#endif
