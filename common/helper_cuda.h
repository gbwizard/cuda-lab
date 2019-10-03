#pragma once

#include <cuda_runtime_api.h>

#include "exceptions.h"

DECLARE_EXCEPTION(InvalidParameterException, common::Exception);
DECLARE_EXCEPTION(CudaErrorException, common::Exception);
DECLARE_EXCEPTION(CudaDeviceException, common::Exception);
DECLARE_EXCEPTION(CudaUnsupportedVersionException, common::Exception);

static const char* cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

void checkCudaError(cudaError_t result, const char* const func, const char* const file, const uint32_t line) {
    if (result) {
        cudaDeviceReset();
        BOOST_THROW_EXCEPTION(CudaErrorException("CUDA error at ") << file << ":" << line << " code=" << result
            << " (" << cudaGetErrorEnum(result) << ") " << "'" << func << "'");
    }
}

#define CHECK_CUDA_ERROR(result) checkCudaError(result, __FUNCTION__, __FILE__, __LINE__)

int32_t ConvertSMVer2Cores(int32_t smMajorVer, int32_t smMinorVer) {
    // SM version uses hex notation: 0x(major)(minor)
    const int32_t smVer = (smMajorVer << 4) + smMinorVer;
    using SM2CoresMap = std::map<int32_t, int32_t>;
    static const SM2CoresMap coresPerSM = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
    };

    const auto coresPerSMIt = coresPerSM.find(smVer);
    if (coresPerSMIt != coresPerSM.end()) {
        return  coresPerSMIt->second;
    }

    BOOST_THROW_EXCEPTION(CudaUnsupportedVersionException("Unsupported SM version: ") << smMajorVer << "." << smMinorVer);
}

int32_t findMaxPerfDevice() {
    int32_t curDevice = 0;
    int32_t smPerMultiproc = 0;
    int32_t maxPerfDevice = 0;
    int32_t devCount = 0;
    int32_t devProhibited = 0;

    uint64_t maxComputePerf = 0;


    CHECK_CUDA_ERROR(cudaGetDeviceCount(&devCount));
    if (! devCount) {
        BOOST_THROW_EXCEPTION(CudaDeviceException("No CUDA device(s) found"));
    }

    // Find the best CUDA capable GPU device
    cudaDeviceProp deviceProp;
    curDevice = 0;
    while (curDevice < devCount) {
        cudaGetDeviceProperties(&deviceProp, curDevice);

        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                smPerMultiproc = 1;
            } else {
                smPerMultiproc = ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
            }

            const uint64_t computePerf = (uint64_t)deviceProp.multiProcessorCount * smPerMultiproc * deviceProp.clockRate;

            if (computePerf > maxComputePerf) {
                maxComputePerf = computePerf;
                maxPerfDevice = curDevice;
            }
        } else {
            devProhibited++;
        }

        ++curDevice;
    }

    if (devProhibited == devCount) {
        BOOST_THROW_EXCEPTION(CudaDeviceException("All devices have compute mode 'Prohibited'"));
    }

    return maxPerfDevice;
}

void initCudaDevice(int32_t devId) {
    if (devId < -1) {
        BOOST_THROW_EXCEPTION(InvalidParameterException("Invalid devId=") << devId);
    }

    if (devId == -1) {
        devId = findMaxPerfDevice();
    }

    CHECK_CUDA_ERROR(cudaSetDevice(devId));

    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, devId));

    std::cout << "GPU" << devId  << " \"" << deviceProp.name << "\": compute capability "
        << deviceProp.major << "." << deviceProp.minor
        << std::endl;
}
