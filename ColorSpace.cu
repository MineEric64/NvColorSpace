#include "ColorSpace.h"
#include <iostream>
#include <fstream>
using namespace std;

#ifdef __CUDACC__
#define CK(grid, block) <<< grid, block >>>
#else
#define CK(grid, block)
#endif

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

__global__ void RGBA32ToBGRA32Kernel(const uint8_t* rgba, uint8_t* bgra, const int offset) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int a = x * 4 + offset;

    bgra[a] = rgba[a + 2];
    bgra[a + 1] = rgba[a + 1];
    bgra[a + 2] = rgba[a];
    bgra[a + 3] = rgba[a + 3];
}

__global__ void RGBA32ToBGR24Kernel(const uint8_t* rgba, uint8_t* bgr, const int offset) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int a = x * 4 + offset;
    int b = x * 3 + offset;

    bgr[b] = rgba[a + 2];
    bgr[b + 1] = rgba[a + 1];
    bgr[b + 2] = rgba[a];
}

__global__ void BGRA32ToYUV420Kernel(const uint8_t* bgra, uint8_t* yuv420, const int offset, const int height, const int frame_size) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int a = x * 4 + offset;
    int j = (x + offset + 1) / height;

    int interpolation = x / 4 + (x % 4) / 2;
    int u_index = frame_size + interpolation;
    int v_index = frame_size + (frame_size / 4) + interpolation;

    int R = bgra[a + 2];
    int G = bgra[a + 1];
    int B = bgra[a];

    int Y = (0.257 * R) + (0.504 * G) + (0.098 * B) + 16.0;
    int U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128.0;
    int V = (0.439 * R) - (0.368 * G) - (0.071 * B) + 128.0;

    Y = MAX(0, MIN(255, Y));
    U = MAX(0, MIN(255, U));
    V = MAX(0, MIN(255, V));

    yuv420[x + offset] = Y;

    if (j % 2 == 0 && x % 2 == 0) {
        //printf("%d]\n", interpolation);
        //printf("[%d, %d]\n", u_index, v_index);
        yuv420[u_index + offset] = U;
        yuv420[v_index + offset] = V;
    }
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

cudaError_t RGBA32ToBGR24(const uint8_t* rgba, uint8_t* bgr, const int width, const int height) {
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

cudaError_t BGRA32ToYUV420(const uint8_t* bgra, uint8_t* yuv420, const int width, const int height) {
    uint8_t* dev_bgra = 0;
    uint8_t* dev_yuv420 = 0;
    cudaError_t cuda_status;
    int size32 = width * height * 4;
    int size_yuv = width * height * 3 / 2;

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

    cuda_status = cudaMalloc((void**)&dev_bgra, size32 * sizeof(uint8_t));

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        if (using_log) log_writer << "cudaMalloc failed!";
        goto Error;
    }

    cuda_status = cudaMalloc((void**)&dev_yuv420, size_yuv * sizeof(uint8_t));

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        if (using_log) log_writer << "cudaMalloc failed!";
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cuda_status = cudaMemcpy(dev_bgra, bgra, size32 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        if (using_log) log_writer << "cudaMemcpy failed!";
        goto Error;
    }

    int thread = 512;
    int frame_size = width * height;
    int block = frame_size / thread;
    int modded = frame_size % thread;

    if (block > 0) BGRA32ToYUV420Kernel CK(block, thread) (dev_bgra, dev_yuv420, 0, height, frame_size);
    if (modded > 0) BGRA32ToYUV420Kernel CK(1, modded) (dev_bgra, dev_yuv420, block, height, frame_size);

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
    cuda_status = cudaMemcpy(yuv420, dev_yuv420, size_yuv * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        if (using_log) log_writer << "cudaMemcpy failed!";
        goto Error;
    }
    else if (using_log) {
        log_writer << "Successfully converted.";
    }

Error:
    cudaFree(dev_bgra);
    cudaFree(dev_yuv420);

    if (using_log) log_writer.close();

    return cuda_status;
}

//uint8_t* RGBAtoYUV420Planar(uint8_t* rgba, int width, int height)
//{
//    const int frameSize = width * height;
//    int yIndex = 0;
//    int vIndex = frameSize;
//    int uIndex = frameSize + (frameSize / 4);
//    int r, g, b, y, u, v;
//    int index = 0;
//
//    uint8_t* buffer = new uint8_t[width * height * 3 / 2];
//
//    for (int j = 0; j < height; j++)
//    {
//        for (int i = 0; i < width; i++)
//        {
//            b = rgba[index * 4 + 0] & 0xff;
//            g = rgba[index * 4 + 1] & 0xff;
//            r = rgba[index * 4 + 2] & 0xff;
//            // a = rgba[index * 4 + 3] & 0xff; unused
//
//            y = (int)(0.257 * r + 0.504 * g + 0.098 * b) + 16;
//            u = (int)(0.439 * r - 0.368 * g - 0.071 * b) + 128;
//            v = (int)(-0.148 * r - 0.291 * g + 0.439 * b) + 128;
//
//            buffer[yIndex++] = (uint8_t)((y < 0) ? 0 : ((y > 255) ? 255 : y));
//
//            if (j % 2 == 0 && index % 2 == 0)
//            {
//                buffer[uIndex++] = (uint8_t)((u < 0) ? 0 : ((u > 255) ? 255 : u));
//                buffer[vIndex++] = (uint8_t)((v < 0) ? 0 : ((v > 255) ? 255 : v));
//            }
//
//            index++;
//        }
//    }
//
//    return buffer;
//}
//
//int main() {
//    uint8_t* bgra = 0;
//    uint8_t* yuv420 = 0;
//
//    bgra = (uint8_t*)malloc(4 * 4 * 4 * sizeof(uint8_t));
//    yuv420 = (uint8_t*)malloc(24 * sizeof(uint8_t));
//
//    for (int i = 0; i < 4; i++) {
//        bgra[0 + i] = 0;
//        bgra[1 + i] = 0;
//        bgra[2+ i] = 255;
//        bgra[3+ i] = 255;
//
//        bgra[4+ i] = 0;
//        bgra[5+ i] = 255;
//        bgra[6+ i] = 0;
//        bgra[7+ i] = 255;
//
//        bgra[8+ i] = 0;
//        bgra[9+ i] = 0;
//        bgra[10+ i] = 255;
//        bgra[11+ i] = 255;
//
//        bgra[12+ i] = 160;
//        bgra[13+ i] = 160;
//        bgra[14+ i] = 160;
//        bgra[15+ i] = 255;
//    }
//
//    BGRA32ToYUV420(bgra, yuv420, 4, 4);
//    //yuv420 = RGBAtoYUV420Planar(bgra, 4, 4);
//
//    for (int i = 0; i < 24; i++) printf("%d, ", yuv420[i]);
//
//    free(bgra);
//    free(yuv420);
//    return 0;
//}