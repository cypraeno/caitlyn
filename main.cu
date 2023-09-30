#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>

#include "vec3.h"
#include "colour.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"


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
 * determine colour of objects hit by a ray to a max 50 depth
 * 
 * the steps are: (1) calculate ray from eye to pixel, (2) determine objects ray intersects, &
 * (3) compute a colour for that intersection point
 * 
 * @param[in] r the ray being shot out from the eye
 * @param[in] world the series of objects in the scene
 * @param[in] local_rand_state the CUDA random state
 * 
 * @returns colour where ray intersects with an object
 * 
 * @relatesalso ray
 */
__device__ colour ray_colour(const ray& r, hittable **world, curandState *local_rand_state) {
    
    ray cur_ray = r;
    float cur_attenuation = 1.0f;
    for (int i = 0; i < 50 ; i++) {
        hit_record rec;
        if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
            vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            cur_attenuation *= 0.5f
            cur_ray = ray(rec.p, target-rec.p)
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            colour c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }

    return colour(0.0, 0.0, 0.0)    // exceeded recursion depth of 50 
}


/**
 * computes random initialization for the renderer
 * 
 * @param[in] max_x the image width in pixels
 * @param[in] max_y the image height in pixels
 * @param[out] rand_state the CUDA random state
 * 
 * @note can also be initialized at the top of render() based on preference
*/
__global__ render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;

    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);    
}


/**
 * identify coordinates of each thread in the image (i, j) and writes it to fb[]
 * 
 * @param[out] fb the frame buffer used to store image colour data
 * @param[in] max_x the image width in pixels
 * @param[in] max_y the image height in pixels
 * @param[in] samples the number of samples per pixel
 * @param[in] cam the camera where rays are shot from
 * @param[in] world the series of objects in the scene
 * @param[in] rand_state the CUDA random state
 * 
 * @warning fb should be cudaMallocManaged()
 */
__global__ void render(colour *fb, int max_x, int max_y, 
                      int samples, camera **cam, hittable **world, curandState *rand_state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= max_x) || (y >= max_y)) return;

    int pixel_index = y*max_x + x;

    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);

    for(int s=0; s < samples; s++) {
        float u = float(x + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(y + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u,v);
        col += ray_colour(r, world);
    }
    
    fb[pixel_index] = col/float(samples);
}


/// initializes the scene and a list of objecs in the scene
__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    // ensures function only runs once in kernal
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}


/// deletes the scene and objects inside
__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for (int i = 0; i < 22*22 + 1 + 3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_list;
    delete *d_world;
    delete *d_camera;
}


int main() {

    // image
    const int image_x = 1440;             // image width
    const int image_y = 720;              // image height
    const int image_s = 100;              // image samples
    const int thread_x = 8;               // thread block x dimension
    const int thread_y = 8;               // thread block y dimension

    std::cerr << "Rendering a " << image_x << "x" << image_y << " image with " << image_s << " samples per pixel ";
    std::cerr << "in " << thread_x << "x" << thread_y << " blocks.\n";

    int image_pixels = image_x * image_y;
    
    // allocate frame buffer (FB) on host to hold RGB values for GPU-CPU communication
    colour *fb;
    size_t fb_size = 3 * image_pixels * sizeof(colour);             // each pixel contains 3 float values (RGB)
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));       // typecast &fb as void** due to CUDA documentation

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMallocManaged((void **)&d_rand_state, image_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // second random state used for world generation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // initializes environment objects
    hitable **d_list;
    int num_hitables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    create_world<<<1,1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());                       // tells CPU to wait until kernal is done before beginning render

    // render FB
    clock_t start, stop;
    start = clock();

    dim3 blocks(image_x / thread_x + 1, image_y / thread_y + 1);    // blocks needed is total image pixels / threads per block
    dim3 threads(thread_x, thread_y);                               // thread_x * thread_y threads in a single block

    render_init<<<blocks, threads>>>(fb, image_x, image_y, image_s, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize()); 

    render<<<blocks, threads>>>(fb, image_x, image_y, image_s, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " s.\n";

    // output FB as image
    std::cout << "P3\n" << image_x << ' ' << image_y << "\n255\n";

    // iterate through FB elements in intervals of 3
    for (int j = image_y-1; j >= 0; j--) {
        for (int i = 0; i < image_x; i++) {
            size_t pixel_index = j*image_x + i;
            write_color(std::cout, fb[pixel_index], image_s);
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaDeviceSynchronize());                       // ensure all kernal processes are done before cleaning up
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));                                  // free FB memory

    cudaDeviceReset();                                              // useful for cuda-memcheck --leak-check full
}
