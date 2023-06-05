#include <iostream>
#include <time.h>

#include "vec3.h"
#include "colour.h"


/** 
 * return error if a CUDA error if encountered during runtime
 * 
 * @param[in] val the error value of a CUDA function call
 * 
 * @note checkCudaErrors() used as simplifcation of check_cuda()
 * @note see cudaError_t enum section of https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
 *      for error code documentation
 */
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_T result, char const *const func, const char *const file, int const line) {
    
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ";" << line << " '" << func << "' \n";

        // make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


/**
 * determine colour of objects hit by a ray
 * 
 * the steps are: (1) calculate ray from eye to pixel, (2) determine objects ray intersects, &
 * (3) compute a colour for that intersection point
 * 
 * @param[in] r the ray being shot out from the eye
 * @param[in] world the series of objects in the scene
 * 
 * @returns colour where ray intersects with an object
 * 
 * @relatesalso ray
 */
__device__ colour ray_colour(const ray& r, hittable **world) {
    hit_record rec;

    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f*vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }

    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}


/**
 * identify coordinates of each thread in the image (i, j) and writes it to fb[]
 * 
 * @param[out] fb the frame buffer used to store image colour data
 * @param[in] max_x the image width in pixels
 * @param[in] max_y the image height in pixels
 * @param[in] lower_left_corner the point where the ray is rendered
 * @param[in] horizontal the horizontal increment which the ray is translated
 * @param[in] vertical the vertical increment which the ray is translated
 * @param[in] origin the point where the ray is shot from
 * @param[in] world the series of objects in the scene
 * 
 * @note __global__ keyword used to define KERNAL FUNCTION
 * 
 * @warning fb should be cudaMallocManaged()
 */
__global__ void render(colour *fb, int max_x, int max_y, 
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
                       hittable **world) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= max_x) || (y >= max_y)) return;

    int pixel_index = y*max_x + x;
    float u = float(x) / float(max_x);
    float v = float(y) / float(max_y);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);

    fb[pixel_index] = ray_colour(r, world);
}


/// initializes the scene and a list of objecs in the scene
__global__ void create_world(hitable **d_list, hitable **d_world) {
    // ensures function only runs once in kernal
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list + 1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hitable_list(d_list,2);
    }
}

/// deletes the scene and objects inside
__global__ void free_world(hitable **d_list, hitable **d_world) {
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
}


int main() {

    // image
    const int image_x = 1200;             // image width
    const int image_y = 600;              // image height
    const int thread_x = 8;               // thread block x dimension
    const int thread_y = 8;               // thread block y dimension

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << thread_x << "x" << thread_y << " blocks.\n";

    int image_pixels = image_x * image_y;
    
    // allocate frame buffer (FB) on host to hold RGB values for GPU-CPU communication
    colour *fb;
    size_t fb_size = 3 * image_pixels * sizeof(colour);             // each pixel contains 3 float values (RGB)
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));       // typecast &fb as void** due to CUDA documentation

    // make world of hitables
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    create_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                       // tells CPU to wait until kernal is done before beginning render

    // render FB
    clock_t start, stop;
    start = clock();

    dim3 blocks(image_x / thread_x + 1, image_y / thread_y + 1);    // blocks needed is total image pixels / threads per block
    dim3 threads(thread_x, thread_y);                               // thread_x * thread_y threads in a single block

    render<<<blocks, threads>>>(fb, image_x, image_y, 
                                vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0),
                                d_world);
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                       // tells CPU to wait until kernal is done before accessing fb[]

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " s.\n";

    // output FB as image
    std::cout << "P3\n" << image_x << ' ' << image_y << "\n255\n";

    // iterate through FB elements in intervals of 3
    for (int j = image_y-1; j >= 0; j--) {
        for (int i = 0; i < image_x; i++) {
            size_t pixel_index = j*image_x + i;
            write_color(std::cout, fb[pixel_index]);
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());                       // ensure all kernal processes are done before cleaning up
    free_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));                              // free objects in scene
    checkCudaErrors(cudaFree(d_world));                             // free scene
    checkCudaErrors(cudaFree(fb));                                  // free FB memory

    cudaDeviceReset();                                              // useful for cuda-memcheck --leak-check full
}
