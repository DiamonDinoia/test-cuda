/**
 * original source provided by Eyal Rozenberg
 * link: https://github.com/eyalroz/cuda-api-wrappers
 */
#include <algorithm>
#include <cuda/api.hpp>
#include <iostream>
#include <memory>

#include "vectorAdd.h"

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

void vectorAddMulti(const int i) {
        int numElements = 50000;
        size_t size     = numElements * sizeof(float);
        std::cout << "[Vector addition of " << numElements << " elements]\n";

        // If we could rely on C++14, we would  use std::make_unique
        auto h_A       = std::unique_ptr<float>(new float[numElements]);
        auto h_B       = std::unique_ptr<float>(new float[numElements]);
        auto h_C       = std::unique_ptr<float>(new float[numElements]);

        auto generator = []() { return rand() / (float)RAND_MAX; };
        std::generate(h_A.get(), h_A.get() + numElements, generator);
        std::generate(h_B.get(), h_B.get() + numElements, generator);

        auto device = cuda::device::get(i);
        auto d_A    = cuda::memory::device::make_unique<float[]>(device, numElements);
        auto d_B    = cuda::memory::device::make_unique<float[]>(device, numElements);
        auto d_C    = cuda::memory::device::make_unique<float[]>(device, numElements);

        cuda::memory::copy(d_A.get(), h_A.get(), size);
        cuda::memory::copy(d_B.get(), h_B.get(), size);

        auto launch_config = cuda::launch_config_builder().overall_size(numElements).block_size(256).build();

        std::cout << "CUDA kernel launch with " << launch_config.dimensions.grid.x << " blocks of "
                  << launch_config.dimensions.block.x << " threads each\n";

        auto kernel       = cuda::kernel::get(device, vectorAdd);
        device.launch(kernel, launch_config, d_A.get(), d_B.get(), d_C.get(), numElements);
        device.synchronize();
        cuda::memory::copy(h_C.get(), d_C.get(), size);

        // Verify that the result vector is correct
        for (int i = 0; i < numElements; ++i) {
            if (fabs(h_A.get()[i] + h_B.get()[i] - h_C.get()[i]) > 1e-5) {
                std::cerr << "Result verification failed at element " << i << "\n";
                exit(EXIT_FAILURE);
            }
        }

        std::cout << "Test PASSED\n";
        std::cout << "SUCCESS\n";
    }
