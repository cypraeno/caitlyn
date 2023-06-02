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
__global__ void render(vec3 *fb, int max_x, int max_y) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= max_x) || (y >= max_y)) return;

    int pixel_index = y*max_x + x;
    fb[pixel_index] = vec3(float(x) / max_x, float(y) / max_y, 0.2f)
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
    vec3 *fb;
    size_t fb_size = 3 * image_pixels * sizeof(vec3);           // each pixel contains 3 float values (RGB)
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));    // typecast &fb as void** due to CUDA documentation

    // render FB
    clock_t start, stop;
    start = clock();

    dim3 blocks(image_x / thread_x + 1, image_y / thread_y + 1);    // blocks needed is total image pixels / threads per block
    dim3 threads(thread_x, thread_y);                               // thread_x * thread_y threads in a single block
    render<<<blocks, threads>>>(fb, image_x, image_y);
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
            write_color(std::cout, (colour)fb[pixel_index]);
        }
    }

    checkCudaErrors(cudaFree(fb));      // free FB memory
}
