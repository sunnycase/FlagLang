cmake_minimum_required (VERSION 3.21)

set(SRCS compiler.cpp)

target_sources(nncaseruntime PRIVATE ${SRCS})
target_include_directories(nncaseruntime PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(nncaseruntime PRIVATE nethost::nethost fmt::fmt nlohmann_json::nlohmann_json)
