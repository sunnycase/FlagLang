cmake_minimum_required (VERSION 3.13)

add_subdirectory(loaders)

set(SRCS runtime_module.cpp
         runtime_function.cpp
         runtime_function.run.cpp)

target_sources(nncaseruntime PRIVATE ${SRCS})
