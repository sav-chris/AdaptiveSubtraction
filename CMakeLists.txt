cmake_minimum_required(VERSION 3.6)
project(AdaptiveSubtractionProject)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(ParallelSTL REQUIRED)
find_package(OpenCLICDLoader REQUIRED)
find_package(
    Boost CONFIG COMPONENTS 
    system 
    filesystem 
    unit_test_framework 
    REQUIRED
)

add_library(adaptiveSubtraction STATIC src/AdaptiveBackgroundSubtraction.cpp src/AdaptiveBackgroundSubtraction.hpp)
add_executable(adaptiveSubtraction.Test 
    src/AdaptiveBackgroundSubtraction.hpp 
    src/AdaptiveBackgroundSubtraction.test.cpp 
)

target_link_libraries(adaptiveSubtraction.Test adaptiveSubtraction )
target_link_libraries(
    adaptiveSubtraction.Test
    Boost::filesystem
    Boost::system
    Boost::unit_test_framework
)
#target_link_libraries(adaptiveSubtraction pstl::ParallelSTL)
target_link_libraries(adaptiveSubtraction OpenCL::OpenCL)
target_link_libraries(adaptiveSubtraction ${OpenCV_LIBS} )


add_executable(AdaptiveSubtractionMain 
    src/main.cpp
)
target_link_libraries(AdaptiveSubtractionMain adaptiveSubtraction)
