#ifndef __PSTD_CAMERA_HPP__
#define __PSTD_CAMERA_HPP__

#include "ray.h"

class camera {
public:
	__device__ camera() {
		m_lower_left_corner = vec3(-2.0, -1.0, -1.0);
		m_horizontal = vec3(4.0, 0.0, 0.0);
		m_vertical = vec3(0.0, 2.0, 0.0);
		m_pos = vec3(0.0, 0.0, 0.0);
	}

	__device__ ray get_ray(float u, float v) {
		return ray(m_pos, m_lower_left_corner + u * m_horizontal + v * m_vertical - m_pos);
	}

	vec3 m_pos;
	vec3 m_lower_left_corner;
	vec3 m_horizontal;
	vec3 m_vertical;
};

#endif