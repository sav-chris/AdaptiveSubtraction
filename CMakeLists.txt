cmake_minimum_required(VERSION 3.29)
project(AdaptiveSubtractionProject)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")

find_package(ParallelSTL REQUIRED)
find_package(CGAL REQUIRED)
find_package(
    Boost CONFIG COMPONENTS 
    system 
    filesystem 
    unit_test_framework 
    REQUIRED
)

add_library(adaptiveSubtraction STATIC src/adaptiveSubtraction.cpp src/adaptiveSubtraction.hpp)
add_executable(adaptiveSubtraction.Test 
    src/adaptiveSubtraction.hpp 
    src/adaptiveSubtraction.test.cpp 
)

target_link_libraries(adaptiveSubtraction CGAL::CGAL )
target_link_libraries(adaptiveSubtraction.Test adaptiveSubtraction )
target_link_libraries(
    adaptiveSubtraction.Test
    Boost::filesystem
    Boost::system
    Boost::unit_test_framework
)
target_link_libraries(adaptiveSubtraction pstl::ParallelSTL)


add_executable(AdaptiveSubtractionMain 
    src/main.cpp
)
target_link_libraries(AdaptiveSubtractionMain adaptiveSubtraction)
