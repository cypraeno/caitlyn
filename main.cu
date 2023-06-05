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
 * determine if a ray hits a given sphere
 * 
 * a sphere of a given radius is initialized at center, and ray r is cast at it,
 * returning true if the ray intersects with the sphere, and false otherwise
 * 
 * @param[in] center the center of the sphere
 * @param[in] radius the radius of the sphere
 * @param[in] r the ray being cast
 * 
 * @returns truth value of whether r hits the sphere
 */
__device__ bool hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;

    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;

    float discriminant = b*b - 4.0f*a*c;
    
    return (discriminant > 0.0f);
}


/**
 * determine colour of objects hit by a ray
 * 
 * the steps are: (1) calculate ray from eye to pixel, (2) determine objects ray intersects, &
 * (3) compute a colour for that intersection point
 * 
 * @param[in] r the ray being shot out from the eye
 * 
 * @returns colour where ray intersects with an object
 * 
 * @relatesalso ray
 */
__device__ colour ray_colour(const ray& r) {
    if (hit_sphere(point3(0, 0, -1), 0.5, r))
        return color(1, 0, 0);

    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y()) + 1.0f;
    return (1.0f - t) * colour(1.0, 1.0, 1.0) + t*colour(0.5, 0.7, 1.0);
}


/**
 * identify coordinates of each thread in the image (i, j) and writes it to fb[]
 * 
 * @param[out] fb the frame buffer used to store image colour data
 * @param[in] max_x the image width in pixels
 * @param[in] max_y the image height in pixels
 * 
 * @note __global__ keyword used to define KERNAL FUNCTION
 * 
 * @warning fb should be cudaMallocManaged()
 */
__global__ void render(colour *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= max_x) || (y >= max_y)) return;

    int pixel_index = y*max_x + x;
    float u = float(x) / float(max_x);
    float v = float(y) / float(max_y);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);

    fb[pixel_index] = ray_colour(r);
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
    size_t fb_size = 3 * image_pixels * sizeof(colour);           // each pixel contains 3 float values (RGB)
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));    // typecast &fb as void** due to CUDA documentation

    // render FB
    clock_t start, stop;
    start = clock();

    dim3 blocks(image_x / thread_x + 1, image_y / thread_y + 1);    // blocks needed is total image pixels / threads per block
    dim3 threads(thread_x, thread_y);                               // thread_x * thread_y threads in a single block

    render<<<blocks, threads>>>(fb, image_x, image_y, 
                                vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0));
    
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

    checkCudaErrors(cudaFree(fb));      // free FB memory
}
