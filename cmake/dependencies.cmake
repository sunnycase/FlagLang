
find_package(nethost REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

if (BUILD_TESTING)
    find_package(GTest REQUIRED)
endif ()
