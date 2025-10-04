#ifndef __PSTD_RAY_HPP__
#define __PSTD_RAY_HPP__

#include "vec3.h"

class ray {
public:
	__device__ ray() {}
	__device__ ray(const vec3& origin, const vec3& direction) { m_origin = origin; m_direction = direction; }
	__device__ vec3 origin() const { return m_origin; }
	__device__ vec3 direction() const { return m_direction; }
	__device__ vec3 at(float t) const { return m_origin + t * m_direction; }

	vec3 m_origin;
	vec3 m_direction;
};

#endif
