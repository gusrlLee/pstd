#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
public:
	__device__ ray() {}
	__device__ ray(const vec3& origin, const vec3& direction) { m_o = origin; m_d = direction; }
	__device__ vec3 origin() const { return m_o; }
	__device__ vec3 direction() const { return m_d; }
	__device__ vec3 at(float t) const { return m_o + t * m_d; }

	vec3 m_o;
	vec3 m_d;
};

#endif
