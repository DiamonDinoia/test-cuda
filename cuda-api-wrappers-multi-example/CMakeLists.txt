cmake_minimum_required(VERSION 3.19)
project(test_API_Wrappers_omp VERSION 1 LANGUAGES CXX)

include(CheckLanguage)

option(ENABLE_OMP "enables omp for parallelism" OFF)

# USING CPM TO HANDLE DEPENDENCIES
set(CPM_DOWNLOAD_VERSION 0.34.0)

if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()

if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
            )
endif ()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
        NAME cudawrappers
        GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
        GIT_TAG development
        EXCLUDE_FROM_ALL YES
        GIT_SHALLOW YES
)

if ("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++23 standard")
    set(CMAKE_CXX_STANDARD 23)
elseif ("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++20 standard")
    set(CMAKE_CXX_STANDARD 20)
elseif ("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "Detected support for C++17 standard")
elseif ("cxx_std_14" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 14)
    message(STATUS "Detected support for C++14 standard")
else ()
    message(FATAL_ERROR "Cannot detect CXX_STANDARD of C++14 or newer.")
endif ()

check_language(CUDA)
if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Requested CUDA support, but no CMAKE_CUDA_COMPILER available")
endif ()
enable_language(CUDA)
if ("cuda_std_23" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
    set(CMAKE_CUDA_STANDARD 23)
    message(STATUS "Detected support for CUDA23 standard")
elseif ("cuda_std_20" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
    set(CMAKE_CUDA_STANDARD 20)
    message(STATUS "Detected support for CUDA20 standard")
elseif ("cuda_std_17" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
    set(CMAKE_CUDA_STANDARD 17)
    message(STATUS "Detected support for CUDA17 standard")
elseif ("cuda_std_14" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
    set(CMAKE_CUDA_STANDARD 14)
    message(STATUS "Detected support for CUDA14 standard")
else ()
    message(FATAL_ERROR "Cannot detect CUDA_STANDARD of CUDA14 or newer.")
endif ()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

function(add_compilation_flags variableName)
    set(old_flags ${${variableName}})
    if (NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        string(REPLACE "-pedantic" "-Wno-pedantic" CUDA_flags "${CUDA_flags}")
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${old_flags}>$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_flags}>")
    endif ()
endfunction()

list(APPEND LIBRARIES cuda-api-wrappers::runtime-and-driver)

if (ENABLE_OMP)
    find_package(OpenMP REQUIRED)
    add_compilation_flags(${OpenMP_CXX_FLAGS})
    add_definitions(-DENABLE_OMP=1)
    list(APPEND LIBRARIES OpenMP::OpenMP_CXX)
endif ()

set(GPU_FLAGS --default-stream per-thread)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${GPU_FLAGS}>")

add_executable(test_multi vectorAdd.cu main.cpp vectorAdd.h)
target_link_libraries(test_multi  ${LIBRARIES})
