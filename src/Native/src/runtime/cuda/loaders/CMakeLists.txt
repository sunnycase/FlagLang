cmake_minimum_required (VERSION 3.13)

set(SRCS cuda_loader.cpp)

target_sources(nncaseruntime PRIVATE ${SRCS})
