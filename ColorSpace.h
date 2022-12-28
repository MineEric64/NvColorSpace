#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef CREATEDLL_EXPORTS
#define NV_DECLSPEC __declspec(dllexport)
#else
#define NV_DECLSPEC __declspec(dllimport)
#endif

union BGRA32 {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
};

union RGBA32 {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

extern "C" NV_DECLSPEC cudaError_t RGBA32ToBGRA32(const uint8_t * rgba, uint8_t * bgra, const int width, const int height);
extern "C" NV_DECLSPEC cudaError_t RGBA32ToBGR24(uint8_t * rgba, uint8_t * bgr, int width, int height);