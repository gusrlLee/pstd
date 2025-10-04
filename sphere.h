#ifndef __PSTD_SPHERE_HPP__
#define __PSTD_SPHERE_HPP__

#include "hitable.h"

class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r) : m_center(cen), m_radius(r) {};
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, surface_interaction& si) const;
	vec3 m_center;
	float m_radius;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, surface_interaction& si) const {
	vec3 oc = r.origin() - m_center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - m_radius * m_radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(b * b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			si.t = temp;
			si.p = r.at(si.t);
			si.n = (si.p - m_center) / m_radius;
			return true;
		}
		temp = (-b + sqrt(b * b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			si.t = temp;
			si.p = r.at(si.t);
			si.n = (si.p - m_center) / m_radius;
			return true;
		}
	}
	return false;
}

#endif