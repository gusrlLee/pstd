#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 color(const ray& r, hitable** world) {
	surface_interaction si;
	if ((*world)->hit(r, 0.0, FLT_MAX, si)) {
		return 0.5f * vec3(si.n.x() + 1.0f, si.n.y() + 1.0f, si.n.z() + 1.0f);
	}
	else {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}

__global__ void init(int scn_width, int scn_height, curandState* rand_state) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i >= scn_width) || (j >= scn_height)) return;
	int pixel_index = j * scn_width + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int scn_width, int scn_height, int spp, camera** cam, hitable** world, curandState* rand_state) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i >= scn_width) || (j >= scn_height)) return;
	int pixel_index = j * scn_width + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 result(0, 0, 0);
	for (int s = 0; s < spp; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(scn_width);
		float v = float(j + curand_uniform(&local_rand_state)) / float(scn_height);
		ray r = (*cam)->get_ray(u, v);
		result += color(r, world);
	}

	fb[pixel_index] = result / float(spp);
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(d_list) = new sphere(vec3(0, 0, -1), 0.5);
		*(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
		*d_world = new hitable_list(d_list, 2);
		*d_camera = new camera();
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera **d_camera) {
	delete* (d_list);
	delete* (d_list + 1);
	delete* d_world;
	delete* d_camera;
}

int main() {
	int w = 1200; // image width
	int h = 800;  // image height

	int tx = 8;   // tile width 
	int ty = 8;   // tile height

	int spp = 16; // samples per pixel
	std::cerr << "Rendering a " << w << "x" << h << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = w * h;
	size_t fb_size = num_pixels * sizeof(vec3);
	unsigned char* image = new unsigned char[num_pixels * 3];

	// allocate random state
	curandState* d_rand_state;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

	vec3* fb; // framebuffer
	CHECK_CUDA_ERROR(cudaMallocManaged((void**)&fb, fb_size));

	// make our world of hitables
	hitable** d_list;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_list, 2 * sizeof(hitable*)));
	hitable** d_world;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_world, sizeof(hitable*)));
	camera** d_camera;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_camera, sizeof(camera*)));

	create_world << <1, 1 >> > (d_list, d_world, d_camera);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	clock_t render_start, render_stop;
	render_start = clock();

	// render our scene
	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);

	init << <blocks, threads >> > (w, h, d_rand_state);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	render<<<blocks, threads >>> (fb, w, h, spp, d_camera, d_world, d_rand_state);

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	render_stop = clock();
	std::cerr << "rendering time: " << ((float)(render_stop - render_start)) / CLOCKS_PER_SEC << " seconds.\n";

	// output fb as ppm image
	for (int j = h - 1; j >= 0; j--) {
		for (int i = 0; i < w; i++) {
			size_t pixel_index = j * w + i;
			int ir = int(255.99 * fb[pixel_index].r());
			int ig = int(255.99 * fb[pixel_index].g());
			int ib = int(255.99 * fb[pixel_index].b());
			
			int image_idx = ((h - j - 1) * w + i) * 3;
			image[image_idx + 0] = ir;
			image[image_idx + 1] = ig;
			image[image_idx + 2] = ib;
		}
	}

	// free the world
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaFree(d_list));
	CHECK_CUDA_ERROR(cudaFree(d_world));
	CHECK_CUDA_ERROR(cudaFree(fb));

	// Write image to file
	stbi_write_png("output.png", w, h, 3, image, w * 3);
	delete[] image;

	cudaDeviceReset();
	return 0;
}