cmake_minimum_required(VERSION 3.15)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

if(NOT DEFINED FLAGLANG_VERSION)
    set(FLAGLANG_VERSION "0.1.0")
endif()

if(DEFINED ENV{FLAGLANG_VERSION_SUFFIX})
    set(FLAGLANG_VERSION_SUFFIX $ENV{FLAGLANG_VERSION_SUFFIX})
endif()

if(NOT DEFINED FLAGLANG_VERSION_SUFFIX)
    find_package (Git)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --always --dirty --tag
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if ("v${FLAGLANG_VERSION}" STREQUAL "${GIT_TAG}")
        set(FLAGLANG_VERSION_SUFFIX "")
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --always --dirty
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_DESC
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(FLAGLANG_VERSION_SUFFIX "-${GIT_DESC}")
    endif()
endif()

if (NOT PACKAGE_VERSION)
    set(PACKAGE_VERSION
        "${FLAGLANG_VERSION}${FLAGLANG_VERSION_SUFFIX}")
endif()

set(NNCASE_VERSION "${FLAGLANG_VERSION}")
set(NNCASE_VERSION_SUFFIX "${FLAGLANG_VERSION_SUFFIX}")

project(flaglang
  VERSION ${FLAGLANG_VERSION}
  LANGUAGES C CXX ASM)

option(BUILD_PYTHON_BINDING "Build python binding" ON)
option(BUILD_TESTING "Build test programs" OFF)

set(FLAGLANG_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(cmake/dependencies.cmake)
include(cmake/compile_flags.cmake)

set(FLAGLANG_MAIN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src/Native/include)
set(FLAGLANG_NTT_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/ntt/include)
set(FLAGLANG_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/Native/include)
set(THIRD_PARTY ${CMAKE_CURRENT_LIST_DIR}/third_party)

set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH OFF)
if (APPLE)
    set(CMAKE_MACOSX_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
else ()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${FLAGLANG_MAIN_INCLUDE_DIR})
include_directories(${FLAGLANG_NTT_INCLUDE_DIR})
include_directories(${FLAGLANG_INCLUDE_DIR})

add_subdirectory(src/Native/include/nncase)
add_subdirectory(src/Native/src)

if(BUILD_TESTING)
    enable_testing()
endif()

add_subdirectory(ntt)

# Python binding
if(BUILD_PYTHON_BINDING)
    add_subdirectory(python/flaglang/native)

    find_program(DOTNET_EXECUTABLE dotnet REQUIRED)
    if(NOT DEFINED FLAGLANG_DOTNET_CONFIGURATION)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(FLAGLANG_DOTNET_CONFIGURATION ${CMAKE_BUILD_TYPE})
        else()
            set(FLAGLANG_DOTNET_CONFIGURATION Release)
        endif()
    endif()
    set(FLAGLANG_MANAGED_COMPILER_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/python/nncase)
    file(MAKE_DIRECTORY ${FLAGLANG_MANAGED_COMPILER_DIR})
    add_custom_target(flaglang-managed-compiler ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${FLAGLANG_MANAGED_COMPILER_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${FLAGLANG_MANAGED_COMPILER_DIR}
        COMMAND ${DOTNET_EXECUTABLE} restore
            ${CMAKE_CURRENT_LIST_DIR}/src/Nncase.Compiler/Nncase.Compiler.csproj
            --locked-mode
            -p:Configuration=${FLAGLANG_DOTNET_CONFIGURATION}
            -p:NuGetAudit=false
            -nr:false
            -m:1
        COMMAND ${DOTNET_EXECUTABLE} publish
            ${CMAKE_CURRENT_LIST_DIR}/src/Nncase.Compiler/Nncase.Compiler.csproj
            -c ${FLAGLANG_DOTNET_CONFIGURATION}
            -o ${FLAGLANG_MANAGED_COMPILER_DIR}
            --no-restore
            --no-self-contained
            -nr:false
            -m:1
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        COMMENT "Publishing Nncase.Compiler for the Python package"
        VERBATIM)
endif()

# Install
install(DIRECTORY src/Native/include/nncase
    DESTINATION include
    COMPONENT flaglang-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.td"
    PATTERN "*.inc"
    PATTERN "LICENSE.TXT"
    PATTERN "*.h"
    )
install(DIRECTORY ntt/include/nncase
    DESTINATION include
    COMPONENT flaglang-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.td"
    PATTERN "*.inc"
    PATTERN "LICENSE.TXT"
    PATTERN "*.h"
    )
install(DIRECTORY ${FLAGLANG_INCLUDE_DIR}/nncase
    DESTINATION include
    COMPONENT flaglang-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.td"
    PATTERN "*.inc"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
    PATTERN "*.h"
    )
install(DIRECTORY python/flaglang
    DESTINATION python
    COMPONENT flaglang-python
    FILES_MATCHING
    PATTERN "*.py"
    )

if(BUILD_PYTHON_BINDING)
    install(DIRECTORY ${FLAGLANG_MANAGED_COMPILER_DIR}/
        DESTINATION _C/nncase
        COMPONENT flaglang-python)
endif()
