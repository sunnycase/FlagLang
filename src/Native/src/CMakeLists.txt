cmake_minimum_required (VERSION 3.13)
#find_package(CUDAToolkit REQUIRED)

set(SRCS 
        api.cpp
        dim_value.cpp
        object.cpp
        padding_value.cpp
        paddings_value.cpp
        shape.cpp
        shape_value.cpp
        tensor.cpp
        type.cpp
        value.cpp)

add_library(nncaseruntime SHARED ${SRCS})
set_property(TARGET nncaseruntime PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(compiler)
add_subdirectory(runtime)

add_executable(nncasetest test.cpp)
#add_executable(nncasetest_cudatest cuda_test.cu)
target_link_libraries(nncasetest PRIVATE nncaseruntime)

if(MSVC)
else()
  if(APPLE)
  else()
    target_compile_options(nncasetest PRIVATE -Wno-ignored-attributes)
  endif()
endif()
if (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo" AND MSVC)
    set_target_properties(nncasetest PROPERTIES LINK_FLAGS "/PROFILE")
endif()
add_executable(nncasetest_cli test_cli.cpp)
target_link_libraries(nncasetest_cli PRIVATE nncaseruntime nlohmann_json::nlohmann_json)
add_executable(nncasetest_dynamic_cli test_dynamic_cli.cpp)
target_link_libraries(nncasetest_dynamic_cli PRIVATE nncaseruntime)

install(TARGETS nncaseruntime EXPORT nncaseTargets
        COMPONENT nncase-runtime
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
install(TARGETS nncaseruntime
        COMPONENT flaglang-python
        LIBRARY DESTINATION "_C"
        RUNTIME DESTINATION "_C"
        )
install(EXPORT nncaseTargets DESTINATION lib/cmake/nncase)
configure_file(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/nncaseConfig.cmake.in nncaseConfig.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncaseConfig.cmake DESTINATION lib/cmake/nncase)
