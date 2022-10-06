//
// Created by mbarbone on 10/6/22.
//
#include <cuda/api.hpp>

#include "vectorAdd.h"

int main() {
    if (cuda::device::count() == 0) {
        std::cerr << "No CUDA devices on this system"
                  << "\n";
        exit(EXIT_FAILURE);
    }
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < cuda::device::count(); ++i) {
        vectorAddMulti(i);
    }
    return 0;
}