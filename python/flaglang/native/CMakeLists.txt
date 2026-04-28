cmake_minimum_required (VERSION 3.18)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED)

set(SRCS env_vars.cpp
         ffi.cpp
         hosting.cpp
         ir.cpp
         nvidia.cpp
         ../../src/interpreter.cc)

pybind11_add_module(triton ${SRCS})
target_link_libraries(triton PRIVATE nncaseruntime ${CMAKE_DL_LIBS})
target_include_directories(triton PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/nvidia/include)
if(WIN32)
  set_target_properties(triton PROPERTIES SUFFIX ".pyd")
  set_target_properties(triton PROPERTIES PREFIX "lib")
else()
  set_target_properties(triton PROPERTIES SUFFIX ".so")
  set_target_properties(triton PROPERTIES PREFIX "lib")
endif()

install(TARGETS triton
        COMPONENT flaglang-python
        LIBRARY DESTINATION "_C")
