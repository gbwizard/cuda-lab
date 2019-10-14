#include <cstdio>
#include <cassert>
#include <iostream>
#include <type_traits>
#include <limits>

#include <boost/program_options.hpp>

#include "common/exceptions.h"
#include "common/helper_cuda.h"

#ifndef __VERBOSE__
#   define __VERBOSE__ 0
#endif

namespace po = boost::program_options;

using MatElemType = float;

constexpr const int32_t kBlockSize = 32;
constexpr const int32_t kKernelRadius = 8;
constexpr const int32_t kKernelRadiusX2 = 2 * kKernelRadius;
constexpr const int32_t kKernelWidth = kKernelRadiusX2 + 1;

__constant__ MatElemType d_Kernel[kKernelWidth * kKernelWidth];

DECLARE_EXCEPTION(RuntimeException, common::Exception);

using CudaKernelFunc = void (*)(MatElemType* resData, const MatElemType* inputData, int32_t width, int32_t height);

__global__
void SimpleConvolution2DKernel(MatElemType* __restrict__ resData, const MatElemType* __restrict__ inputData, int32_t width, int32_t height)
{
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    const int32_t dx = bx * blockDim.x + tx;
    const int32_t dy = by * blockDim.y + ty;

    if (dx >= width || dy >= height) {
        return;
    }

    MatElemType resultElem{0}, inputElem{0};
    int32_t inputX, inputY;
    for (int32_t i = -kKernelRadius; i <= kKernelRadius; i++)	{ // by row
        if (dy + i < 0) {
            inputY = 0;
        } else if (dy + i < height) {
            inputY = dy + i;
        } else {
            inputY = height - 1;
        }
        for (int32_t j = -kKernelRadius; j <= kKernelRadius; j++) { // by column
            if (dx + j < 0) {
                inputX = 0;
            } else if (dx + j < width) {
                inputX = dx + j;
            } else {
                inputX = width - 1;
            }

            inputElem = inputData[inputY * width + inputX];
            const int32_t kx = j + kKernelRadius;
            const int32_t ky = i + kKernelRadius;
            resultElem += inputElem * d_Kernel[ky * kKernelWidth + kx];
        }
    }

    resData[dy * width + dx] = resultElem;
}

__global__
void SimpleConvolution2DRowKernel(MatElemType* __restrict__ resData, const MatElemType* __restrict__ inputData, int32_t width, int32_t height)
{
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    const int32_t dx = bx * blockDim.x + tx;
    const int32_t dy = by * blockDim.y + ty;

    if (dx >= width || dy >= height) {
        return;
    }

    MatElemType resultElem{0}, inputElem{0};
    int32_t inputX;
    for (int32_t j = -kKernelRadius; j <= kKernelRadius; j++) { // by column
        if (dx + j < 0) {
            inputX = 0;
        } else if (dx + j < width) {
            inputX = dx + j;
        } else {
            inputX = width - 1;
        }

        inputElem = inputData[dy * width + inputX];
        const int32_t kx = j + kKernelRadius;
        resultElem += inputElem * d_Kernel[kx];
    }

    resData[dy * width + dx] = resultElem;
}

__global__
void SimpleConvolution2DColKernel(MatElemType* __restrict__ resData, const MatElemType* __restrict__ inputData, int32_t width, int32_t height)
{
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    const int32_t dx = bx * blockDim.x + tx;
    const int32_t dy = by * blockDim.y + ty;

    if (dx >= width || dy >= height) {
        return;
    }

    MatElemType resultElem{0}, inputElem{0};
    int32_t inputY;
    for (int32_t i = -kKernelRadius; i <= kKernelRadius; i++) { // by column
        if (dy + i < 0) {
            inputY = 0;
        } else if (dy + i < height) {
            inputY = dy + i;
        } else {
            inputY = width - 1;
        }

        inputElem = inputData[inputY * width + dx];
        const int32_t kx = i + kKernelRadius;
        resultElem += inputElem * d_Kernel[kx];
    }

    resData[dy * width + dx] = resultElem;
}

__global__
void SharedConvolution2DKernel(MatElemType* __restrict__ resData, const MatElemType* __restrict__ inputData, int32_t width, int32_t height)
{
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    const int32_t dx = bx * blockDim.x + tx;
    const int32_t dy = by * blockDim.y + ty;

    if (dx >= width || dy >= height) {
        return;
    }

    constexpr const int32_t kMaxCacheWidth = kBlockSize + kKernelRadiusX2;
    constexpr const int32_t kMaxCacheHeight = kBlockSize + kKernelRadiusX2;
    __shared__ MatElemType blockCache[kMaxCacheWidth * kMaxCacheHeight];
    const int32_t kBlockXSize = ((bx + 1) * kBlockSize > width) ? (width - bx * kBlockSize) : kBlockSize;
    const int32_t kBlockYSize = ((by + 1) * kBlockSize > height) ? (height - by * kBlockSize) : kBlockSize;
    const int32_t kCacheWidth = ((bx + 1) * kBlockSize > width + kKernelRadiusX2) ? (width + kKernelRadiusX2 - bx * kBlockSize) : kBlockSize + kKernelRadiusX2;
    const int32_t kCacheHeight = ((by + 1) * kBlockSize > height + kKernelRadiusX2) ? (height + kKernelRadiusX2 - by * kBlockSize) : kBlockSize + kKernelRadiusX2;

    int32_t dataX, dataY;
    const int32_t cache2InputXoff = bx * blockDim.x - kKernelRadius;
    const int32_t cache2InputYoff = by * blockDim.y - kKernelRadius;
    for (int32_t i = ty; i < kCacheHeight; i += kBlockYSize) {  // by row
        if (i + cache2InputYoff < 0) { // top apron
            dataY = 0;
        } else if (i + cache2InputYoff < height) { // inside data
            dataY = i + cache2InputYoff;
        } else { // bottom apron
            dataY = height - 1;
        }
        const int32_t dataRowOff = dataY * width;
        const int32_t cacheRowOff = i * kCacheWidth;
        for (int32_t j = tx; j < kCacheWidth; j += kBlockXSize) { // by column
            if (j + cache2InputXoff < 0) { // left apron
                dataX = 0;
            } else if (j + cache2InputXoff < width) { // inside data
                dataX = j + cache2InputXoff;
            } else { // right apron
                dataX = width - 1;
            }
            blockCache[cacheRowOff + j] = inputData[dataRowOff + dataX];
        }
    }

    // Sync all threads in block after memory load
    __syncthreads();

    MatElemType resultElem{0}, cacheElem{0};
    for (int32_t i = -kKernelRadius; i <= kKernelRadius; i++)	{ // by row
        const int32_t ky = i + kKernelRadius;
        const int32_t cacheRowOff = (ty + kKernelRadius + i) * kCacheWidth;
        const int32_t kernelRowOff = ky * kKernelWidth;
        for (int32_t j = -kKernelRadius; j <= kKernelRadius; j++) { // by column
            cacheElem = blockCache[cacheRowOff + tx + kKernelRadius + j];
            const int32_t kx = j + kKernelRadius;
            resultElem += cacheElem * d_Kernel[kernelRowOff + kx];
        }
    }

    resData[dy * width + dx] = resultElem;
}

__global__
void SharedConvolution2DFastKernel(MatElemType* __restrict__ resData, const MatElemType* __restrict__ inputData, int32_t width, int32_t height)
{
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    const int32_t dx = bx * blockDim.x + tx;
    const int32_t dy = by * blockDim.y + ty;

    constexpr const int32_t kCacheWidth = kBlockSize + kKernelRadiusX2;
    constexpr const int32_t kCacheHeight = kBlockSize + kKernelRadiusX2;
    __shared__ MatElemType blockCache[kCacheWidth * kCacheHeight];

    int32_t dataX, dataY;
    const int32_t cache2InputXoff = bx * blockDim.x - kKernelRadius;
    const int32_t cache2InputYoff = by * blockDim.y - kKernelRadius;
    for (int32_t i = ty; i < kCacheHeight; i += kBlockSize) {  // by row
        if (i + cache2InputYoff < 0) { // top apron
            dataY = 0;
        } else if (i + cache2InputYoff < height) { // inside data
            dataY = i + cache2InputYoff;
        } else { // bottom apron
            dataY = height - 1;
        }
        const int32_t dataRowOff = dataY * width;
        const int32_t cacheRowOff = i * kCacheWidth;
        for (int32_t j = tx; j < kCacheWidth; j += kBlockSize) { // by column
            if (j + cache2InputXoff < 0) { // left apron
                dataX = 0;
            } else if (j + cache2InputXoff < width) { // inside data
                dataX = j + cache2InputXoff;
            } else { // right apron
                dataX = width - 1;
            }
            blockCache[cacheRowOff + j] = inputData[dataRowOff + dataX];
        }
    }

    // Sync all threads in block after memory load
    __syncthreads();

    MatElemType resultElem{0}, cacheElem{0};

#pragma unroll
    for (int32_t i = -kKernelRadius; i <= kKernelRadius; i++)	{ // by row
        const int32_t ky = i + kKernelRadius;
        const int32_t cacheRowOff = (ty + kKernelRadius + i) * kCacheWidth;
        const int32_t kernelRowOff = ky * kKernelWidth;

#pragma unroll
        for (int32_t j = -kKernelRadius; j <= kKernelRadius; j++) { // by column
            cacheElem = blockCache[cacheRowOff + tx + kKernelRadius + j];
            const int32_t kx = j + kKernelRadius;
            resultElem += cacheElem * d_Kernel[kernelRowOff + kx];
        }
    }

    resData[dy * width + dx] = resultElem;
}

__global__
void SharedConvolution2DRowKernel(MatElemType* __restrict__ resData, const MatElemType* __restrict__ inputData, int32_t width, int32_t height)
{
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    const int32_t dx = bx * blockDim.x + tx;
    const int32_t dy = by * blockDim.y + ty;

    if (dx >= width || dy >= height) {
        return;
    }

    constexpr const int32_t kMaxCacheWidth = kBlockSize + kKernelRadiusX2;
    constexpr const int32_t kMaxCacheHeight = kBlockSize;
    __shared__ MatElemType blockCache[kMaxCacheWidth * kMaxCacheHeight];
    const int32_t kBlockXSize = ((bx + 1) * kBlockSize > width) ? (width - bx * kBlockSize) : kBlockSize;
    const int32_t kCacheWidth = ((bx + 1) * kBlockSize > width + kKernelRadiusX2) ? (width + kKernelRadiusX2 - bx * kBlockSize) : kBlockSize + kKernelRadiusX2;

    const int32_t cache2InputXoff = bx * blockDim.x - kKernelRadius;
    int32_t dataX;
    const int32_t dataRowOff = dy * width;
    const int32_t cacheRowOff = ty * kCacheWidth;
    for (int32_t j = tx; j < kCacheWidth; j += kBlockXSize) { // by column
        if (j + cache2InputXoff < 0) { // left apron
            dataX = 0;
        } else if (j + cache2InputXoff < width) { // inside data
            dataX = j + cache2InputXoff;
        } else { // right apron
            dataX = width - 1;
        }
        blockCache[cacheRowOff + j] = inputData[dataRowOff + dataX];
    }

    // Sync all threads in block after memory load
    __syncthreads();

    MatElemType resultElem{0}, cacheElem{0};

#pragma unroll
    for (int32_t j = -kKernelRadius; j <= kKernelRadius; j++) { // by column
        cacheElem = blockCache[cacheRowOff + tx + kKernelRadius + j];
        const int32_t kx = j + kKernelRadius;
        resultElem += cacheElem * d_Kernel[kx];
    }

    resData[dy * width + dx] = resultElem;
}

__global__
void SharedConvolution2DColKernel(MatElemType* __restrict__ resData, const MatElemType* __restrict__ inputData, int32_t width, int32_t height)
{
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    const int32_t dx = bx * blockDim.x + tx;
    const int32_t dy = by * blockDim.y + ty;

    if (dx >= width || dy >= height) {
        return;
    }

    constexpr const int32_t kMaxCacheWidth = kBlockSize;
    constexpr const int32_t kMaxCacheHeight = kBlockSize + kKernelRadiusX2;
    __shared__ MatElemType blockCache[kMaxCacheWidth * kMaxCacheHeight];
    const int32_t kBlockYSize = ((by + 1) * kBlockSize > height) ? (height - by * kBlockSize) : kBlockSize;
    const int32_t kCacheHeight = ((by + 1) * kBlockSize > height + kKernelRadiusX2) ? (height + kKernelRadiusX2 - by * kBlockSize) : kBlockSize + kKernelRadiusX2;

    const int32_t cache2InputYoff = by * blockDim.y - kKernelRadius;
    int32_t dataY;
    for (int32_t i = ty; i < kCacheHeight; i += kBlockYSize) { // by column
        if (i + cache2InputYoff < 0) { // left apron
            dataY = 0;
        } else if (i + cache2InputYoff < height) { // inside data
            dataY = i + cache2InputYoff;
        } else { // right apron
            dataY = height - 1;
        }
        blockCache[i * kBlockSize + tx] = inputData[dataY * width + dx];
    }

    // Sync all threads in block after memory load
    __syncthreads();

    MatElemType resultElem{0}, cacheElem{0};

#pragma unroll
    for (int32_t i = -kKernelRadius; i <= kKernelRadius; i++) { // by column
        cacheElem = blockCache[(i + ty + kKernelRadius) * kBlockSize + tx];
        const int32_t kx = i + kKernelRadius;
        resultElem += cacheElem * d_Kernel[kx];
    }

    resData[dy * width + dx] = resultElem;
}

#if __VERBOSE__
constexpr const int32_t kRowBlockDimX = 1;
constexpr const int32_t kRowBlockDimY = 1;
constexpr const int32_t kRowDataSteps = 1;
constexpr const int32_t kRowApronSteps = 1;
#else
constexpr const int32_t kRowBlockDimX = 8;
constexpr const int32_t kRowBlockDimY = 2;
constexpr const int32_t kRowDataSteps = 16;
constexpr const int32_t kRowApronSteps = 1;
#endif

__global__
void SharedConvolution2DRowFastKernel(MatElemType* __restrict__ resData, const MatElemType* __restrict__ inputData, int32_t width, int32_t height)
{
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    constexpr const int32_t kCacheWidth = kRowBlockDimX * (kRowDataSteps + 2 * kRowApronSteps);
    constexpr const int32_t kCacheHeight = kRowBlockDimY;
    __shared__ MatElemType blockCache[kCacheHeight][kCacheWidth];

    const int32_t baseX = (bx * kRowDataSteps - kRowApronSteps) * kRowBlockDimX + tx;
    const int32_t baseY = by * kRowBlockDimY + ty;
    inputData += baseY * width + baseX;
    resData   += baseY * width + baseX;

    // Load left apron
    const MatElemType leftApronVal = inputData[kRowApronSteps * kRowBlockDimX - tx];
#pragma unroll
    for (int32_t xStep = 0; xStep < kRowApronSteps; ++xStep) {
        blockCache[ty][xStep * kRowBlockDimX + tx] = (baseX >= -xStep * kRowBlockDimX)
            ? inputData[xStep * kRowBlockDimX]
            : leftApronVal;
    }

    // Load right apron
    const MatElemType rightApronVal = inputData[(kRowApronSteps + kRowDataSteps) * kRowBlockDimX - 1 - tx];
#pragma unroll
    for (int32_t xStep = kRowApronSteps + kRowDataSteps; xStep < kRowApronSteps + kRowDataSteps + kRowApronSteps; ++xStep) {
        blockCache[ty][xStep * kRowBlockDimX + tx] = (width - baseX > xStep * kRowBlockDimX)
            ? inputData[xStep * kRowBlockDimX] : rightApronVal;
    }

    // Load main data
#pragma unroll
    for (int32_t xStep = kRowApronSteps; xStep < kRowApronSteps + kRowDataSteps; ++xStep) {
        blockCache[ty][xStep * kRowBlockDimX + tx] = inputData[xStep * kRowBlockDimX];
    }

    // Sync all threads in block after memory load
    __syncthreads();

#pragma unroll
    for (int32_t xStep = kRowApronSteps; xStep < kRowApronSteps + kRowDataSteps; ++xStep) {
        MatElemType resultElem{0};

#pragma unroll
        for (int32_t j = -kKernelRadius; j <= kKernelRadius; j++) { // by column
            resultElem += d_Kernel[j + kKernelRadius] * blockCache[ty][xStep * kRowBlockDimX + tx + j];
        }

        resData[xStep * kRowBlockDimX] = resultElem;
    }
}

#if __VERBOSE__
constexpr const int32_t kColBlockDimX = 1;
constexpr const int32_t kColBlockDimY = 1;
constexpr const int32_t kColDataSteps = 1;
constexpr const int32_t kColApronSteps = 1;
#else
constexpr const int32_t kColBlockDimX = 32;
constexpr const int32_t kColBlockDimY = 8;
constexpr const int32_t kColDataSteps = 8;
constexpr const int32_t kColApronSteps = 1;
#endif

__global__
void SharedConvolution2DColFastKernel(MatElemType* __restrict__ resData, const MatElemType* __restrict__ inputData, int32_t width, int32_t height)
{
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    constexpr const int32_t kCacheWidth = kColBlockDimX;
    constexpr const int32_t kCacheHeight = kColBlockDimY * (kColDataSteps + 2 * kColApronSteps);
    __shared__ MatElemType blockCache[kCacheWidth][kCacheHeight + 1];

    const int32_t baseX = bx * kColBlockDimX + tx;
    const int32_t baseY = (by * kColDataSteps - kColApronSteps) * kColBlockDimY + ty;
    inputData += baseY * width + baseX;
    resData   += baseY * width + baseX;

    // Load top apron
    const MatElemType topApronVal = inputData[(kColApronSteps * kColBlockDimY - ty) * width];
#pragma unroll
    for (int32_t yStep = 0; yStep < kColApronSteps; ++yStep) {
        blockCache[tx][yStep * kColBlockDimY + ty] = (baseY >= -yStep * kColBlockDimY)
            ? inputData[yStep * kColBlockDimY * width]
            : topApronVal;
    }

    // Load bottom apron
    const MatElemType bottomApronVal = inputData[((kColApronSteps + kColDataSteps) * kColBlockDimY - ty - 1) * width];
#pragma unroll
    for (int32_t yStep = kColApronSteps + kColDataSteps; yStep < kColApronSteps + kColDataSteps + kColApronSteps; ++yStep) {
        blockCache[tx][yStep * kColBlockDimY + ty] = (height - baseY > yStep * kColBlockDimY)
            ? inputData[yStep * kColBlockDimY * width]
            : bottomApronVal;
    }

    // Load main data
#pragma unroll
    for (int32_t yStep = kColApronSteps; yStep < kColApronSteps + kColDataSteps; ++yStep) {
        blockCache[tx][yStep * kColBlockDimY + ty] = inputData[yStep * kColBlockDimY * width];
    }

    // Sync all threads in block after memory load
    __syncthreads();

#pragma unroll
    for (int32_t yStep = kColApronSteps; yStep < kColApronSteps + kColDataSteps; ++yStep) {
        MatElemType resultElem{0};

#pragma unroll
        for (int32_t j = -kKernelRadius; j <= kKernelRadius; j++) {
            resultElem += d_Kernel[j + kKernelRadius] * blockCache[tx][yStep * kColBlockDimY + ty + j];
        }

        resData[yStep * kColBlockDimY * width] = resultElem;
    }
}

template<typename TFloat>
void RandomInit(std::vector<TFloat>& data, typename std::enable_if<std::is_floating_point<TFloat>::value>::type* dummy = 0)
{
    std::srand(uint32_t(std::time(0)));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<TFloat>(static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX));
    }
}

template<typename TInt>
void RandomInit(std::vector<TInt>& data, typename std::enable_if<std::is_integral<TInt>::value>::type* dummy = 0)
{
    std::srand(uint32_t(std::time(0)));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<MatElemType>(std::rand() % 10);
    }
}

void PrintMat(const std::vector<MatElemType>& data, int32_t width, int32_t height) {
    for (int32_t i = 0; i < height; ++i) {
        for (int32_t j = 0; j < width; ++j) {
            const int32_t elemIdx = i * width + j;
            std::cout << " " << data[elemIdx];
        }
        std::cout << std::endl;
    }
}

int32_t divCeil(int32_t a, int32_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Reference row convolution filter for separable convolution
void CpuConvolution2DSepRows(std::vector<MatElemType>& h_Dst, const std::vector<MatElemType>& h_Src,
    const std::vector<MatElemType>& h_Kernel, int32_t imageW, int32_t imageH)
{
    for (int32_t y = 0; y < imageH; y++) {
        for (int32_t x = 0; x < imageW; x++) {
            MatElemType sum = 0;
            for (int32_t k = -kKernelRadius; k <= kKernelRadius; k++) {
                int32_t d = x + k;

                if (d < 0)
                    d = 0;

                if (d >= imageW)
                    d = imageW - 1;

                sum += h_Src[y * imageW + d] * h_Kernel[k + kKernelRadius];
            }

            h_Dst[y * imageW + x] = sum;
        }
    }
}

// Reference column convolution filter for separable convolution
void CpuConvolution2DSepColumns(std::vector<MatElemType>& h_Dst, const std::vector<MatElemType>& h_Src,
    const std::vector<MatElemType>& h_Kernel, int32_t imageW, int32_t imageH)
{
    for (int32_t y = 0; y < imageH; y++) {
        for (int32_t x = 0; x < imageW; x++) {
            MatElemType sum = 0;
            for (int32_t k = -kKernelRadius; k <= kKernelRadius; k++) {
                int32_t d = y + k;

                if (d < 0)
                    d = 0;

                if (d >= imageH)
                    d = imageH - 1;

                sum += h_Src[d * imageW + x] * h_Kernel[k + kKernelRadius];
            }

            h_Dst[y * imageW + x] = sum;
        }
    }
}

void CpuConvolution2DResult(std::vector<MatElemType>& result, const std::vector<MatElemType>& input, int32_t inputWidth, int32_t inputHeight,
                         const std::vector<MatElemType>& kernel)
{
    MatElemType resultElem{0}, inputElem{0};
    int32_t inputX, inputY;
    for (int32_t dy = 0; dy < inputHeight; ++dy) {
        const int32_t resRowOff = dy * inputWidth;
        for (int32_t dx = 0; dx < inputWidth; ++dx) {
            for (int32_t i = -kKernelRadius; i <= kKernelRadius; ++i)	{ // by row
                if (dy + i < 0) {
                    inputY = 0;
                } else if (dy + i < inputHeight) {
                    inputY = dy + i;
                } else {
                    inputY = inputHeight - 1;
                }
                const int32_t inputRowOff = inputY * inputWidth;
                const int32_t ky = i + kKernelRadius;
                const int32_t kernelRowOff = ky * kKernelWidth;
                for (int32_t j = -kKernelRadius; j <= kKernelRadius; ++j) { // by column
                    if (dx + j < 0) {
                        inputX = 0;
                    } else if (dx + j < inputWidth) {
                        inputX = dx + j;
                    } else {
                        inputX = inputWidth - 1;
                    }
                    inputElem = input[inputRowOff + inputX];
                    const int32_t kx = j + kKernelRadius;
                    resultElem += inputElem * kernel[kernelRowOff + kx];
                }
            }
            result[resRowOff + dx] = resultElem;
            resultElem = 0;
        }
    }
}

int32_t CpuConvolution2D(const std::vector<MatElemType>& input, int32_t inputWidth, int32_t inputHeight,
                         const std::vector<MatElemType>& kernel)
{
    std::vector<MatElemType> result(input.size());
    CpuConvolution2DResult(result, input, inputWidth, inputHeight, kernel);
    return EXIT_SUCCESS;
}

int32_t CpuConvolution2DSep(const std::vector<MatElemType>& input, int32_t inputWidth, int32_t inputHeight,
                         const std::vector<MatElemType>& kernel)
{
    std::vector<MatElemType> intermed(input.size());
    CpuConvolution2DSepRows(intermed, input, kernel, inputWidth, inputHeight);
    std::vector<MatElemType> result(input.size());
    CpuConvolution2DSepColumns(result, intermed, kernel, inputWidth, inputHeight);
    return EXIT_SUCCESS;
}

int32_t GpuConvolution2D(const std::vector<MatElemType>& input, int32_t inputWidth, int32_t inputHeight
                            , const std::vector<MatElemType>& kernel, CudaKernelFunc cudaKernelFunc)
{
    // Allocate device memory
    MatElemType* d_Input, *d_Result;
    const uint32_t memSizeInput = input.size() * sizeof(MatElemType);
    const uint32_t memSizeKernel = kernel.size() * sizeof(MatElemType);
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_Input), memSizeInput));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_Result), memSizeInput));

    // Copy host memory to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_Input, input.data(), memSizeInput, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), memSizeKernel, 0, cudaMemcpyHostToDevice));

    // Setup execution parameters
    const dim3 cudaBlock(kBlockSize, kBlockSize);
    const dim3 cudaGrid(divCeil(inputWidth, cudaBlock.x), divCeil(inputHeight, cudaBlock.y));
    std::cout << "CUDA blocks: " << cudaGrid.x << "x" << cudaGrid.y << std::endl;

    std::cout << "Convolution using CUDA Kernel: warm up ..." << std::endl;
    // Performs warmup operation using matrixMul CUDA kernel
    cudaKernelFunc<<<cudaGrid, cudaBlock>>>(d_Result, d_Input, inputWidth, inputHeight);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "Convolution using CUDA Kernel: warm up ... Done" << std::endl;

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Execute the kernel
    const int32_t nIter = 100;
    std::cout << "Convolution using CUDA Kernel (" << nIter << " times) ..." << std::endl;
    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, NULL));
    for (int j = 0; j < nIter; j++) {
        cudaKernelFunc<<<cudaGrid, cudaBlock>>>(d_Result, d_Input, inputWidth, inputHeight);
    }
    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    std::cout << "Convolution using CUDA Kernel (" << nIter << " times) ... Done" << std::endl;

    float msecTotal = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    const float msecPerRun = msecTotal / nIter;
    std::cout << "Time=" << msecPerRun << " msec/run, Threads=" << cudaBlock.x * cudaBlock.y << " thread/block" << std::endl;

    // Copy result from device to host
    std::vector<MatElemType> gpuResult(input.size());
    CHECK_CUDA_ERROR(cudaMemcpy(gpuResult.data(), d_Result, memSizeInput, cudaMemcpyDeviceToHost));

    std::cout << "Checking computed result for correctness ..." << std::endl;
    std::vector<MatElemType> refResult(input.size());
    CpuConvolution2DResult(refResult, input, inputWidth, inputHeight, kernel);

#if __VERBOSE__
    std::cout << "Input:" << std::endl;
    PrintMat(input, inputWidth, inputHeight);
    std::cout << "Kernel:" << std::endl;
    PrintMat(kernel, kKernelWidth, kKernelWidth);
    std::cout << "GpuResult:" << std::endl;
    PrintMat(gpuResult, inputWidth, inputHeight);
    std::cout << "RefResult:" << std::endl;
    PrintMat(refResult, inputWidth, inputHeight);
#endif

    bool correct = true;
    // Test relative error by the formula |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    const double kEps = 1.e-6;  // machine zero
    for (size_t i = 0; i < gpuResult.size(); i++) {
        const double absErr = std::fabs(static_cast<double>(gpuResult[i] - refResult[i]));
        const double dotLength = inputWidth;
        const double absVal = std::fabs(static_cast<double>(gpuResult[i]));
        const double relErr = absErr / absVal / dotLength;

        if (relErr > kEps) {
            std::cerr << "Error! GpuResult[" << i << "]=" << static_cast<double>(gpuResult[i])
                      << ", RefResult[" << i << "]=" << refResult[i] << ", err>" << kEps << std::endl;
            correct = false;
            break;
        }
    }

    std::cout << "Checking computed result for correctness ... " << (correct ? "PASS" : "FAIL") << std::endl;

    // Clean up device memory
    CHECK_CUDA_ERROR(cudaFree(d_Input));
    CHECK_CUDA_ERROR(cudaFree(d_Result));

    return (correct ? EXIT_SUCCESS : EXIT_FAILURE);
}

int32_t GpuConvolution2DSep(const std::vector<MatElemType>& input, int32_t inputWidth, int32_t inputHeight
                            , const std::vector<MatElemType>& kernel
                            , CudaKernelFunc cudaKernelRowFunc, const dim3& rowCudaBlock, const dim3& rowCudaGrid
                            , CudaKernelFunc cudaKernelColFunc, const dim3& colCudaBlock, const dim3& colCudaGrid)
{
    // Allocate device memory
    MatElemType* d_Input, *d_Intermed, *d_Result;
    const uint32_t memSizeInput = input.size() * sizeof(MatElemType);
    const uint32_t memSizeKernel = kernel.size() * sizeof(MatElemType);
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_Input), memSizeInput));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_Intermed), memSizeInput));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_Result), memSizeInput));

    // Copy host memory to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_Input, input.data(), memSizeInput, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), memSizeKernel, 0, cudaMemcpyHostToDevice));

    std::cout << "Rows CUDA blocks:     " << rowCudaGrid.x << "x" << rowCudaGrid.y << std::endl;
    std::cout << "Rows CUDA threads:    " << rowCudaBlock.x << "x" << rowCudaBlock.y << std::endl;
    std::cout << "Columns CUDA blocks:  " << colCudaGrid.x << "x" << colCudaGrid.y << std::endl;
    std::cout << "Columns CUDA threads: " << colCudaBlock.x << "x" << colCudaBlock.y << std::endl;

    std::cout << "Convolution using CUDA Kernel: warm up ..." << std::endl;
    // Performs warmup operation using matrixMul CUDA kernel
    cudaKernelRowFunc<<<rowCudaGrid, rowCudaBlock>>>(d_Intermed, d_Input, inputWidth, inputHeight);
    cudaKernelColFunc<<<colCudaGrid, colCudaBlock>>>(d_Result, d_Intermed, inputWidth, inputHeight);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "Convolution using CUDA Kernel: warm up ... Done" << std::endl;

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Execute the kernel
    const int32_t nIter = 100;
    std::cout << "Convolution using CUDA Kernel (" << nIter << " times) ..." << std::endl;
    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, NULL));
    for (int32_t j = 0; j < nIter; j++) {
        cudaKernelRowFunc<<<rowCudaGrid, rowCudaBlock>>>(d_Intermed, d_Input, inputWidth, inputHeight);
        cudaKernelColFunc<<<colCudaGrid, colCudaBlock>>>(d_Result, d_Intermed, inputWidth, inputHeight);
    }
    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    std::cout << "Convolution using CUDA Kernel (" << nIter << " times) ... Done" << std::endl;

    float msecTotal = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    const float msecPerRun = msecTotal / nIter;
    std::cout << "Time: " << msecPerRun << " msec/run" << std::endl;
    std::cout << "Throughput: " << std::setprecision(std::numeric_limits<double>::digits10) << static_cast<double>(input.size()) / msecPerRun * 0.001 << " MElems/sec" << std::endl;

    // Copy result from device to host
    std::vector<MatElemType> gpuResult(input.size());
    CHECK_CUDA_ERROR(cudaMemcpy(gpuResult.data(), d_Result, memSizeInput, cudaMemcpyDeviceToHost));

    std::cout << "Checking computed result for correctness ..." << std::endl;
    std::vector<MatElemType> refIntermed(input.size());
    std::vector<MatElemType> refResult(input.size());
    CpuConvolution2DSepRows(refIntermed, input, kernel, inputWidth, inputHeight);
    CpuConvolution2DSepColumns(refResult, refIntermed, kernel, inputWidth, inputHeight);

#if __VERBOSE__
    std::vector<MatElemType> gpuIntermed(input.size());
    CHECK_CUDA_ERROR(cudaMemcpy(gpuIntermed.data(), d_Intermed, memSizeInput, cudaMemcpyDeviceToHost));

    std::cout << "Input:" << std::endl;
    PrintMat(input, inputWidth, inputHeight);
    std::cout << "Kernel:" << std::endl;
    PrintMat(kernel, kKernelWidth, 1);
    std::cout << "GpuIntermed:" << std::endl;
    PrintMat(gpuIntermed, inputWidth, inputHeight);
    std::cout << "GpuResult:" << std::endl;
    PrintMat(gpuResult, inputWidth, inputHeight);
    std::cout << "RefIntermed:" << std::endl;
    PrintMat(refIntermed, inputWidth, inputHeight);
    std::cout << "RefResult:" << std::endl;
    PrintMat(refResult, inputWidth, inputHeight);

    {
        // Test relative error by the formula |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
        const double kEps = 1.e-6;  // machine zero
        for (size_t i = 0; i < gpuIntermed.size(); i++) {
            const double absErr = std::fabs(static_cast<double>(gpuIntermed[i] - refIntermed[i]));
            const double dotLength = inputWidth;
            const double absVal = std::fabs(static_cast<double>(gpuIntermed[i]));
            const double relErr = absErr / absVal / dotLength;

            if (relErr > kEps) {
                std::cerr << "Error! GpuIntermed[" << i << "]=" << static_cast<double>(gpuIntermed[i])
                          << ", RefIntermed[" << i << "]=" << refIntermed[i] << ", err>" << kEps << std::endl;
                break;
            }
        }
    }
#endif

    bool correct = true;
    // Test relative error by the formula |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    const double kEps = 1.e-6;  // machine zero
    for (size_t i = 0; i < gpuResult.size(); i++) {
        const double absErr = std::fabs(static_cast<double>(gpuResult[i] - refResult[i]));
        const double dotLength = inputWidth;
        const double absVal = std::fabs(static_cast<double>(gpuResult[i]));
        const double relErr = absErr / absVal / dotLength;

        if (relErr > kEps) {
            std::cerr << "Error! GpuResult[" << i << "]=" << static_cast<double>(gpuResult[i])
                      << ", RefResult[" << i << "]=" << refResult[i] << ", err>" << kEps << std::endl;
            correct = false;
            break;
        }
    }

    std::cout << "Checking computed result for correctness ... " << (correct ? "PASS" : "FAIL") << std::endl;

    // Clean up device memory
    CHECK_CUDA_ERROR(cudaFree(d_Input));
    CHECK_CUDA_ERROR(cudaFree(d_Intermed));
    CHECK_CUDA_ERROR(cudaFree(d_Result));

    return (correct ? EXIT_SUCCESS : EXIT_FAILURE);
}

int main(int argc, char **argv) {
    // Program options
    int32_t devId = -1;
    int32_t width = 0;
    int32_t height = 0;
    std::string modeStr;
    bool isSeparable = false;

    po::options_description options("Options");
    options.add_options()
        ("help,h", "display this message")
        ("device,d", po::value<int32_t>(&devId), "CUDA device to use")
        ("mode,m", po::value<std::string>(&modeStr)->default_value("shared"), "Mode (cpu, simple, shared, sharedfast)")
        ("separable,s", "Separable convolution")
        ("width,w", po::value<int32_t>(&width)->default_value(1024), "Width of input data")
        ("height,t", po::value<int32_t>(&height)->default_value(1024), "Height of input data")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, options), vm);
        po::notify(vm);
    } catch(const po::error& ex) {
        std::cerr << "Failed to parse command line options: " << ex.what() << std::endl;
        std::cout << options << std::endl;
        return EXIT_FAILURE;
    }

    if (vm.count("separable")) {
        isSeparable = true;
    }

    const int2 dimInputSize{width, height};
    const uint32_t elemCount = dimInputSize.x * dimInputSize.y;
    std::vector<MatElemType> inputData(elemCount);
    RandomInit(inputData);

    std::cout << "Applying convolution kernel " << kKernelWidth << "x" << kKernelWidth << " to data " <<  width << "x" << height << std::endl;
    std::cout << "Separable kernel: " << (isSeparable ? "Yes" : "NO") << std::endl;
    std::cout << "Using mode: " << modeStr << std::endl;

    const uint32_t kernelElemCount = isSeparable ? kKernelWidth : kKernelWidth * kKernelWidth;
    std::vector<MatElemType> kernelData(kernelElemCount);
    RandomInit(kernelData);

    try {
        initCudaDevice(devId);
    } catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << ". Bailout" << std::endl;
        return EXIT_FAILURE;
    }

    int32_t status = EXIT_SUCCESS;
    if (modeStr == "cpu") {
        if (isSeparable) {
            status = CpuConvolution2DSep(inputData, width, height, kernelData);
        } else {
            status = CpuConvolution2D(inputData, width, height, kernelData);
        }
    } else if (modeStr == "simple") {
        if (isSeparable) {
            const dim3 rowCudaBlock(kRowBlockDimX, kRowBlockDimY);
            const dim3 rowCudaGrid(divCeil(width, rowCudaBlock.x), divCeil(height, rowCudaBlock.y));
            const dim3 colCudaBlock(kColBlockDimX, kColBlockDimY);
            const dim3 colCudaGrid(divCeil(width, colCudaBlock.x), divCeil(height, colCudaBlock.y));
            status = GpuConvolution2DSep(inputData, width, height, kernelData, SimpleConvolution2DRowKernel, rowCudaBlock, rowCudaGrid
                                         , SimpleConvolution2DColKernel, colCudaBlock, colCudaGrid);
        } else {
            status = GpuConvolution2D(inputData, width, height, kernelData, SimpleConvolution2DKernel);
        }
    } else if (modeStr == "shared") {
        if (isSeparable) {
            const dim3 rowCudaBlock(kRowBlockDimX, kRowBlockDimY);
            const dim3 rowCudaGrid(divCeil(width, rowCudaBlock.x), divCeil(height, rowCudaBlock.y));
            const dim3 colCudaBlock(kColBlockDimX, kColBlockDimY);
            const dim3 colCudaGrid(divCeil(width, colCudaBlock.x), divCeil(height, colCudaBlock.y));
            status = GpuConvolution2DSep(inputData, width, height, kernelData, SharedConvolution2DRowKernel, rowCudaBlock, rowCudaGrid
                                         , SharedConvolution2DColKernel, colCudaBlock, colCudaGrid);
        } else {
            status = GpuConvolution2D(inputData, width, height, kernelData, SharedConvolution2DKernel);
        }
    } else if (modeStr == "sharedfast") {
        static_assert(kRowBlockDimX * kRowApronSteps >= kKernelRadius, "Too little space for left/right apron");
        static_assert(kColBlockDimY * kColApronSteps >= kKernelRadius, "Too little space for top/bottom apron");
        if (width % (kRowDataSteps * kRowBlockDimX) != 0) {
            std::cout << "Width (" << width << ") is not divisable by kRowDataSteps * kRowBlockDimX (" << kRowDataSteps * kRowBlockDimX << "). Bailout" << std::endl;
            return EXIT_FAILURE;
        }
        if (height % kRowBlockDimY != 0) {
            std::cout << "Height (" << height << ") is not divisable by kRowBlockDimY (" << kRowBlockDimY << "). Bailout" << std::endl;
            return EXIT_FAILURE;
        }
        if (width % kRowBlockDimX != 0) {
            std::cout << "Width (" << width << ") is not divisable by kColBlockDimX (" << kColBlockDimX << "). Bailout" << std::endl;
            return EXIT_FAILURE;
        }
        if (height % kRowBlockDimY != 0) {
            std::cout << "Height (" << height << ") is not divisable by kColDataSteps * kColBlockDimY (" << kColDataSteps * kColBlockDimY << "). Bailout" << std::endl;
            return EXIT_FAILURE;
        }
        if (isSeparable) {
            const dim3 rowCudaBlock(kRowBlockDimX, kRowBlockDimY);
            const dim3 rowCudaGrid(width / (rowCudaBlock.x * kRowDataSteps), height / rowCudaBlock.y);
            const dim3 colCudaBlock(kColBlockDimX, kColBlockDimY);
            const dim3 colCudaGrid(width / colCudaBlock.x, height / (colCudaBlock.y * kColDataSteps));
            status = GpuConvolution2DSep(inputData, width, height, kernelData, SharedConvolution2DRowFastKernel, rowCudaBlock, rowCudaGrid
                                         , SharedConvolution2DColFastKernel, colCudaBlock, colCudaGrid);
        } else {
            status = GpuConvolution2D(inputData, width, height, kernelData, SharedConvolution2DFastKernel);
        }
    } else {
        std::cerr << "Unsupported mode: " << modeStr << ". Bailout" << std::endl;
        return EXIT_FAILURE;
    }

    return status;
}
