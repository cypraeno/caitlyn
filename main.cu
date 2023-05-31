#include <iostream>
#include <time.h>

// checkCudaErrors(val) returns error if a CUDA error if encountered during runtime
// effects: may produce output to std:cerr stream
// notes: #define statement to simplify function call from check_cuda() to checkCudaErrors()
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_T result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ";" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// render(gb, max_x, max_y) identifies coordinates of each thread in the image (i, j) and writes it to fb[]
// requires: fb[] is cudaMallocManaged()
//           max_x, max_y are width and height of rendered image in pixels
// effects: may mutate fb[]
// notes: __global__ keyword used to define KERNAL FUNCTION
__global__ void render(float *fb, int max_x, int max_y) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= max_x) || (y >= max_y)) return;

    int pixel_index = y*max_x*3 + x*3;
    fb[pixel_index + 0] = float(x) / max_x;
    fb[pixel_index + 1] = float(y) / max_y;
    fb[pixel_index + 2] = 0.2;
}


int main() {


    // image
    const int image_x = 1200;             // image width
    const int image_y = 600;              // image height
    const int thread_x = 8;               // thread block x dimension
    const int thread_y = 8;               // thread block y dimension

    std:cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std:cerr << "in " << thread_x << "x" << thread_y << " blocks.\n";

    int image_pixels = image_x * image_y;
    

    // allocate frame buffer (FB) on host to hold RGB values for GPU-CPU communication
    float *fb;
    size_t fb_size = 3 * image_pixels * sizeof(float)           // each pixel contains 3 float values (RGB)
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size))    // typecast &fb as void** due to CUDA documentation


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
    std:cerr << "took " << timer_seconds << " s.\n";


    // output FB as image
    std::cout << "P3\n" << image_x << ' ' << image_y << "\n255\n";

    for (int j = image_y-1; j >= 0; j--) {

        for (int i = 0; i < image_x; i++) {

            size_t pixel_index = j*3*image_x + i*3;

            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    checkCudaErrors(cudaFree(fb));      // free FB memory
}