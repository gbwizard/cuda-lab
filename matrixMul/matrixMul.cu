#include <cstdio>
#include <cassert>
#include <iostream>

#include <boost/program_options.hpp>

#include "common/exceptions.h"
#include "common/helper_cuda.h"

namespace po = boost::program_options;

using MatElemType = float;

const int32_t kBlockSize = 32;

DECLARE_EXCEPTION(RuntimeException, common::Exception);

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
__global__
void SharedMatrixMulKernel(MatElemType* C, const MatElemType* const A, const MatElemType* const B, int32_t wA, int32_t hA, int32_t wB, int32_t hB) {
    // Block index
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    // Thread index
    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    const int32_t aBegin = wA * kBlockSize * by;

    // Index of the last sub-matrix of A processed by the block
    const int32_t aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    const int32_t aStep = kBlockSize;

    // Index of the first sub-matrix of B processed by the block
    const int32_t bBegin = kBlockSize * bx;

    // Step size used to iterate through the sub-matrices of B
    const int32_t bStep = kBlockSize * wB;

    // Csub is used to store the element of the block sub-matrix that is computed by the thread
    MatElemType Csub = 0;

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to store the sub-matrix of A
        __shared__ MatElemType As[kBlockSize][kBlockSize];

        // Declaration of the shared memory array Bs used to store the sub-matrix of B
        __shared__ MatElemType Bs[kBlockSize][kBlockSize];

        const int32_t ax = a - aBegin + tx;
        const int32_t ay = by * kBlockSize + ty;
        const int32_t ai = a + wA * ty + tx;

        const int32_t bmx = bx * kBlockSize + tx;
        const int32_t bmy = (b - bBegin) / wB + ty;
        const int32_t bi = b + wB * ty + tx;

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        As[ty][tx] = (ax < wA && ay < hA) ? A[ai] : static_cast<MatElemType>(0);
        Bs[ty][tx] = (bmx < wB && bmy < hB) ? B[bi] : static_cast<MatElemType>(0);

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together; each thread computes one element of the block sub-matrix
#pragma unroll
        for (int32_t k = 0; k < kBlockSize; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding computation is done before loading two new  sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    const int32_t c = wB * kBlockSize * by + kBlockSize * bx;
    const int32_t ci = c + wB * ty + tx;
    const int32_t cx = bx * kBlockSize + tx;
    const int32_t cy = by * kBlockSize + ty;
    if (cx < wB && cy < hA) {
        C[ci] = Csub;
    }
}

__global__
void SimpleMatrixMulKernel(MatElemType *C, const MatElemType* const A, const MatElemType* const B, int32_t wA, int32_t hA, int32_t wB)
{
    // Block index
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    // Thread index
    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;

    const int32_t i = blockDim.y * by + ty;
    const int32_t j = blockDim.x * bx + tx;

    if (i >= hA || j >= wB) {
        return;
    }

    MatElemType res{0};
    for (int32_t r = 0; r < wA; ++r) {
        const MatElemType a = A[wA * i + r];
        const MatElemType b = B[wB * r + j];
        res += a * b;
    }
    C[wB * i + j] = res;
}

void RandomInit(MatElemType* data, int32_t size)
{
    std::srand(uint32_t(std::time(0)));
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<MatElemType>(static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX));
    }
}

int SharedMatrixMultiply(const dim3& dimsA, const dim3& dimsB) {
    // Allocate host memory for matrices A and B
    const uint32_t elemCountA = dimsA.x * dimsA.y;
    const uint32_t memSizeA = sizeof(MatElemType) * elemCountA;
    MatElemType* h_A = reinterpret_cast<MatElemType*>(malloc(memSizeA));
    const uint32_t elemCountB = dimsB.x * dimsB.y;
    const uint32_t memSizeB = sizeof(MatElemType) * elemCountB;
    MatElemType* h_B = reinterpret_cast<MatElemType*>(malloc(memSizeB));

    // Initialize host memory
    RandomInit(h_A, elemCountA);
    RandomInit(h_B, elemCountB);

    // Allocate device memory
    MatElemType *d_A, *d_B, *d_C;

    // Allocate host matrix C
    const dim3 dimsC(dimsB.x, dimsA.y, 1);
    const uint32_t elemCountC = dimsC.x * dimsC.y;
    const uint32_t memSizeC = elemCountC * sizeof(MatElemType);
    MatElemType* h_C = reinterpret_cast<MatElemType *>(malloc(memSizeC));
    MatElemType* h_Ccheck = reinterpret_cast<MatElemType *>(malloc(memSizeC));

    if (! h_C) {
        std::cerr << "Failed to allocate host matrix C" << std::endl;
        exit(EXIT_FAILURE);
    }

    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_A), memSizeA));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_B), memSizeB));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_C), memSizeC));

    // copy host memory to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, memSizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, memSizeB, cudaMemcpyHostToDevice));

    // Setup execution parameters
    const dim3 cudaBlock(kBlockSize, kBlockSize);
    auto divCeilInt = [] (int32_t a, int32_t b) -> int32_t {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    };
    const dim3 cudaGrid(divCeilInt(dimsB.x, cudaBlock.x), divCeilInt(dimsA.y, cudaBlock.y));
    std::cout << "CUDA blocks: " << cudaGrid.x << "x" << cudaGrid.y << std::endl;

    // Performs warmup operation using matrixMul CUDA kernel
    std::cout << "Computing result using CUDA Kernel: warm up ..." << std::endl;
    SharedMatrixMulKernel<<<cudaGrid, cudaBlock>>>(d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    std::cout << "Computing result using CUDA Kernel: warm up ... Done" << std::endl;

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Execute the kernel
    const int32_t nIter = 10;
    std::cout << "Computing result using CUDA Kernel (" << nIter << " times) ..." << std::endl;
    CHECK_CUDA_ERROR(cudaEventRecord(start, NULL));
    for (int32_t j = 0; j < nIter; j++) {
        SharedMatrixMulKernel<<<cudaGrid, cudaBlock>>>(d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, NULL));
    std::cout << "Computing result using CUDA Kernel (" << nIter << " times) ... Done" << std::endl;

    // Wait for the stop event to complete
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    const float msecPerMatrixMul = msecTotal / nIter;
    const double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
    const double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    std::cout << "Performance=" <<  gigaFlops << " GFlop/s, Time=" << msecPerMatrixMul << " msec, FlopsPerMatMul="
        << flopsPerMatrixMul << " Ops, WorkgroupSize=" << cudaBlock.x * cudaBlock.y << " threads/block"
        << std::endl;

    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, memSizeC, cudaMemcpyDeviceToHost));

    std::cout << "Checking computed result for correctness ..." << std::endl;

    for (uint32_t i = 0; i < dimsA.y; ++i) {
        for (uint32_t j = 0; j < dimsB.x; ++j) {
            MatElemType& elemC = h_Ccheck[i * dimsC.x + j] = MatElemType{0};
            for (uint32_t r = 0; r < dimsA.x; ++r) {
                const MatElemType elemA = h_A[dimsA.x * i + r];
                const MatElemType elemB = h_B[dimsB.x * r + j];
                elemC += elemA * elemB;
            }
        }
    }

    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    const double kEps = 1.e-6;  // machine zero
    for (int32_t i = 0; i < elemCountC; i++) {
        const double abs_err = std::fabs(static_cast<double>(h_C[i] - h_Ccheck[i]));
        const double dot_length = dimsA.x;
        const double abs_val = std::fabs(static_cast<double>(h_C[i]));
        const double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > kEps) {
            std::cerr << "Error! C[" << i << "]=" << std::setprecision(15) << static_cast<double>(h_C[i]) << ", Cref[" << i << "]="
                << static_cast<double>(h_Ccheck[i]) << ", err>" << kEps
                << std::endl;
            correct = false;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Checking computed result for correctness ... " << (correct ? "PASS" : "FAIL") << std::endl;

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Ccheck);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return (correct ? EXIT_SUCCESS : EXIT_FAILURE);
}

void RandomInit(std::vector<MatElemType>& data)
{
    std::srand(uint32_t(std::time(0)));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<MatElemType>(static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX));
    }
}

int32_t SimpleMatrixMultiply(const dim3& dimsA, const dim3& dimsB) {
    // Allocate host memory for matrices A and B
    const uint32_t elemCountA = dimsA.x * dimsA.y;
    std::vector<MatElemType> hostA(elemCountA);
    const uint32_t elemCountB = dimsB.x * dimsB.y;
    std::vector<MatElemType> hostB(elemCountB);

    // Initialize host memory with random elements
    RandomInit(hostA);
    RandomInit(hostB);

    // Allocate host matrix C
    const dim3 dimsC(dimsB.x, dimsA.y, 1);
    const uint32_t elemCountC = dimsC.x * dimsC.y;
    std::vector<MatElemType> hostC(elemCountC);

    // Allocate device memory
    MatElemType *d_A, *d_B, *d_C;
    const uint32_t memSizeA = elemCountA * sizeof(MatElemType);
    const uint32_t memSizeB = elemCountB * sizeof(MatElemType);
    const uint32_t memSizeC = elemCountC * sizeof(MatElemType);
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_A), memSizeA));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_B), memSizeB));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_C), memSizeC));

    // copy host memory to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, hostA.data(), memSizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, hostB.data(), memSizeB, cudaMemcpyHostToDevice));

    // Setup execution parameters
    const dim3 cudaBlock(kBlockSize, kBlockSize);

    auto divCeilInt = [] (int32_t a, int32_t b) -> int32_t {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    };
    const dim3 cudaGrid(divCeilInt(dimsB.x, cudaBlock.x), divCeilInt(dimsA.y, cudaBlock.y));
    std::cout << "CUDA blocks: " << cudaGrid.x << "x" << cudaGrid.y << std::endl;

    std::cout << "Computing result using CUDA Kernel: warm up ..." << std::endl;
    // Performs warmup operation using matrixMul CUDA kernel
    SimpleMatrixMulKernel<<<cudaGrid, cudaBlock>>>(d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "Computing result using CUDA Kernel: warm up ... Done" << std::endl;

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Execute the kernel
    const int32_t nIter = 10;
    std::cout << "Computing result using CUDA Kernel (" << nIter << " times) ..." << std::endl;
    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, NULL));
    for (int j = 0; j < nIter; j++) {
        SimpleMatrixMulKernel<<<cudaGrid, cudaBlock>>>(d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x);
    }
    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    std::cout << "Computing result using CUDA Kernel (" << nIter << " times) ... Done" << std::endl;

    float msecTotal = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    const float msecPerMatrixMul = msecTotal / nIter;
    const double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
    const double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    std::cout << "Performance=" << gigaFlops << " GFlop/s, "
               << "TimePerMatMul=" << msecPerMatrixMul << " msec, "
               << "FlopsPerMatMul=" << flopsPerMatrixMul << " flops, "
               << "ThreadsInBlock=" << cudaBlock.x * cudaBlock.y << " thread/block"
               << std::endl;

    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(hostC.data(), d_C, memSizeC, cudaMemcpyDeviceToHost));

    std::cout << "Checking computed result for correctness ..." << std::endl;
    std::vector<MatElemType> checkHostC(elemCountC);
    for (uint32_t i = 0; i < dimsA.y; ++i) {
        for (uint32_t j = 0; j < dimsB.x; ++j) {
            MatElemType& elemC = checkHostC[i * dimsC.x + j] = MatElemType{0};
            for (uint32_t r = 0; r < dimsA.x; ++r) {
                const MatElemType elemA = hostA[dimsA.x * i + r];
                const MatElemType elemB = hostB[dimsB.x * r + j];
                elemC += elemA * elemB;
            }
        }
    }

    bool correct = true;

    // Test relative error by the formula |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    const double eps = 1.e-6;  // machine zero

    for (size_t i = 0; i < hostC.size(); i++) {
        const double absErr = std::fabs(static_cast<double>(hostC[i] - checkHostC[i]));
        const double dotLength = dimsA.x;
        const double absVal = std::fabs(static_cast<double>(hostC[i]));
        const double relErr = absErr / absVal / dotLength;

        if (relErr > eps) {
            std::cerr << "Error! C[" << i << "]=" << static_cast<double>(hostC[i]) << ", Cref[" << i << "]=" << static_cast<double>(checkHostC[i]) << ", err>" << eps << std::endl;
            correct = false;
            break;
        }
    }

    std::cout << "Checking computed result for correctness ... " << (correct ? "PASS" : "FAIL") << std::endl;

    // Clean up device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return (correct ? EXIT_SUCCESS : EXIT_FAILURE);
}

int main(int argc, char **argv) {
    // Program options
    int32_t devId = -1;
    uint32_t wa = 0;
    uint32_t ha = 0;
    uint32_t wb = 0;
    uint32_t hb = 0;
    std::string modeStr{"shared"};

    po::options_description options("Options");
    options.add_options()
        ("help,h", "display this message")
        ("device,d", po::value<int32_t>(&devId), "CUDA device to use")
        ("mode,m", po::value<std::string>(&modeStr)->default_value("shared"), "Mode (shared, simple)")
        ("wa", po::value<uint32_t>(&wa)->default_value(25 * 2 * kBlockSize), "Width of Matrix A (MUST be equal to height of Matrix B)")
        ("ha", po::value<uint32_t>(&ha)->default_value(25 * 2 * kBlockSize), "Height of Matrix A")
        ("wb", po::value<uint32_t>(&wb)->default_value(25 * 4 * kBlockSize), "Width of Matrix B")
        ("hb", po::value<uint32_t>(&hb)->default_value(25 * 2 * kBlockSize), "Height of Matrix B (MUST be equal to width of Matrix A)")
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

    if (wa != hb) {
        std::cerr << "Width of Matrix A (" << wa << ") != heigth of Matrix B (" << ha << "). Bailout" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Multiplying A[" << ha << "][" << wa << "] x B[" << hb << "][" << wb <<"] using mode '" << modeStr << "'" << std::endl;

    try {
        initCudaDevice(devId);
    } catch (const std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << ". Bailout" << std::endl;
        return EXIT_FAILURE;
    }

    const dim3 dimsA(wa, ha, 1);
    const dim3 dimsB(wb, hb, 1);

    int32_t multStatus = EXIT_SUCCESS;
    if (modeStr == "simple") {
        multStatus = SimpleMatrixMultiply(dimsA, dimsB);
    } else if (modeStr == "shared") {
        multStatus = SharedMatrixMultiply(dimsA, dimsB);
    } else {
        std::cerr << "Unsupported mode: " << modeStr << ". Bailout" << std::endl;
        return EXIT_FAILURE;
    }

    return multStatus;
}

