cmake_minimum_required(VERSION 3.19)
project(test_API_Wrappers VERSION 1 LANGUAGES CXX)

include(CheckLanguage)

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

enable_language(CUDA)

CPMAddPackage(
    NAME cudawrappers
    GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
    GIT_TAG v0.5.5
    EXCLUDE_FROM_ALL YES
    GIT_SHALLOW YES
)

add_executable(test_executable vectorAdd.cu)
target_link_libraries(test_executable cuda-api-wrappers::runtime-and-driver)
set_property(TARGET test_executable PROPERTY CXX_STANDARD 14)
