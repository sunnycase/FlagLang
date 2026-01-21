cmake_minimum_required (VERSION 3.13)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(loaders)

set(SRCS runtime_module.cpp
         runtime_function.cpp
         runtime_function.run.cpp)

target_sources(nncaseruntime PRIVATE ${SRCS})
target_link_libraries(nncaseruntime PUBLIC CUDA::cudart)
