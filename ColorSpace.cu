#include "ColorSpace.h"
#include <iostream>
#include <fstream>
using namespace std;

#ifdef __CUDACC__
#define CK(grid, block) <<< grid, block >>>
#else
#define CK(grid, block)
#endif

__global__ void RGBA32ToBGRA32Kernel(const uint8_t* rgba, uint8_t* bgra, int offset) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int a = x * 4 + offset;

    bgra[a] = rgba[a + 2];
    bgra[a + 1] = rgba[a + 1];
    bgra[a + 2] = rgba[a];
    bgra[a + 3] = rgba[a + 3];
}

__global__ void RGBA32ToBGR24Kernel(uint8_t* rgba, uint8_t* bgr, int offset) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int a = x * 4 + offset;
    int b = x * 3 + offset;

    bgr[b] = rgba[a + 2];
    bgr[b + 1] = rgba[a + 1];
    bgr[b + 2] = rgba[a];
}

cudaError_t RGBA32ToBGRA32(const uint8_t* rgba, uint8_t* bgra, const int width, const int height) {
    uint8_t* dev_rgba = 0;
    uint8_t* dev_bgra = 0;
    cudaError_t cuda_status;
    int size = width * height * 4;

    bool using_log = false;
    ofstream log_writer;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        if (using_log) log_writer << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
        goto Error;
    }

    // Allocate GPU buffers for three vectors (one input, one output) 
    cuda_status = cudaMalloc((void**)&dev_rgba, size * sizeof(uint8_t));

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        if (using_log) log_writer << "cudaMalloc failed!";
        goto Error;
    }

    cuda_status = cudaMalloc((void**)&dev_bgra, size * sizeof(uint8_t));

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        if (using_log) log_writer << "cudaMalloc failed!";
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cuda_status = cudaMemcpy(dev_rgba, rgba, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        if (using_log) log_writer << "cudaMemcpy failed!";
        goto Error;
    }

    int thread = 512;
    int block = (width * height) / thread;
    int modded = (width * height) % thread;

    if (block > 0) RGBA32ToBGRA32Kernel CK(block, thread) (dev_rgba, dev_bgra, 0);
    if (modded > 0) RGBA32ToBGRA32Kernel CK(1, modded) (dev_rgba, dev_bgra, block);

    // Check for any errors launching the kernel
    cuda_status = cudaGetLastError();

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        if (using_log) log_writer << "addKernel launch failed: " << cudaGetErrorString(cuda_status) << '\n';
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda_status = cudaDeviceSynchronize();

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cuda_status);
        if (using_log) log_writer << "cudaDeviceSynchronize returned error code " << cuda_status << " after launching addKernel!\n";
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cuda_status = cudaMemcpy(bgra, dev_bgra, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        if (using_log) log_writer << "cudaMemcpy failed!";
        goto Error;
    }
    else if (using_log) {
        log_writer << "Successfully converted.";
    }

Error:
    cudaFree(dev_rgba);
    cudaFree(dev_bgra);

    if (using_log) log_writer.close();

    return cuda_status;
}

cudaError_t RGBA32ToBGR24(uint8_t* rgba, uint8_t* bgr, int width, int height) {
    uint8_t* dev_rgba = 0;
    uint8_t* dev_bgr = 0;
    cudaError_t cuda_status;
    int size24 = width * height * 3;
    int size32 = width * height * 4;

    bool using_log = false;
    ofstream log_writer;

    if (using_log) log_writer = ofstream("BetterNvLog.log");

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        if (using_log) log_writer << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
        goto Error;
    }

    cuda_status = cudaMalloc((void**)&dev_rgba, size32 * sizeof(uint8_t));

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        if (using_log) log_writer << "cudaMalloc failed!";
        goto Error;
    }

    cuda_status = cudaMalloc((void**)&dev_bgr, size24 * sizeof(uint8_t));

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        if (using_log) log_writer << "cudaMalloc failed!";
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cuda_status = cudaMemcpy(dev_rgba, rgba, size32 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        if (using_log) log_writer << "cudaMemcpy failed!";
        goto Error;
    }

    int thread = 512;
    int block = (width * height) / thread;
    int modded = (width * height) % thread;

    if (block > 0) RGBA32ToBGR24Kernel CK(block, thread) (dev_rgba, dev_bgr, 0);
    if (modded > 0) RGBA32ToBGR24Kernel CK(1, modded) (dev_rgba, dev_bgr, block);

    // Check for any errors launching the kernel
    cuda_status = cudaGetLastError();

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        if (using_log) log_writer << "addKernel launch failed: " << cudaGetErrorString(cuda_status) << '\n';
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda_status = cudaDeviceSynchronize();

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cuda_status);
        if (using_log) log_writer << "cudaDeviceSynchronize returned error code " << cuda_status << " after launching addKernel!\n";
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cuda_status = cudaMemcpy(bgr, dev_bgr, size24 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        if (using_log) log_writer << "cudaMemcpy failed!";
        goto Error;
    }
    else if (using_log) {
        log_writer << "Successfully converted.";
    }

Error:
    cudaFree(dev_rgba);
    cudaFree(dev_bgr);

    if (using_log) log_writer.close();

    return cuda_status;
}