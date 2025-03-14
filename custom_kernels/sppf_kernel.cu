// #include <cuda_runtime.h>
// #include <float.h>

// __global__ void maxPoolKernel(
//     const float* input, float* output, 
//     int batch, int channels, int height, int width,
//     int kernel_size, int pad)
// {
//     int tx = threadIdx.x; // Thread x within block
//     int ty = threadIdx.y; // Thread y within block
//     int bx = blockIdx.x;  // Block x in grid
//     int by = blockIdx.y;  // Block y in grid
//     int bz = blockIdx.z;  // Batch index

//     // Tile dimensions (adjust based on shared memory size)
//     const int TILE_SIZE = 32;
//     extern __shared__ float shmem[];

//     // Output coordinates
//     int out_x = bx * (TILE_SIZE - kernel_size + 1) + tx;
//     int out_y = by * (TILE_SIZE - kernel_size + 1) + ty;
//     int c = blockIdx.z % channels; // Channel index
//     int b = bz / channels;         // Batch index

//     if (b >= batch || c >= channels) return;

//     // Load tile into shared memory (with halo for kernel)
//     int halo = kernel_size - 1;
//     for (int i = ty; i < TILE_SIZE + halo; i += blockDim.y) {
//         for (int j = tx; j < TILE_SIZE + halo; j += blockDim.x) {
//             int in_y = by * (TILE_SIZE - kernel_size + 1) - pad + i;
//             int in_x = bx * (TILE_SIZE - kernel_size + 1) - pad + j;
//             float val = -FLT_MAX;
//             if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
//                 int idx = ((b * channels + c) * height + in_y) * width + in_x;
//                 val = input[idx];
//             }
//             if (i < TILE_SIZE + halo && j < TILE_SIZE + halo) {
//                 shmem[i * (TILE_SIZE + halo) + j] = val;
//             }
//         }
//     }
//     __syncthreads();

//     // Compute max over k × k window
//     if (out_x < width && out_y < height) {
//         float max_val = -FLT_MAX;
//         for (int ky = 0; ky < kernel_size; ky++) {
//             for (int kx = 0; kx < kernel_size; kx++) {
//                 float val = shmem[(ty + ky) * (TILE_SIZE + halo) + (tx + kx)];
//                 max_val = fmaxf(max_val, val);
//             }
//         }
//         int out_idx = ((b * channels + c) * height + out_y) * width + out_x;
//         output[out_idx] = max_val;
//     }
// }