#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include "AdaptiveBackgroundSubtraction.hpp"


int main() 
{
    return 0;
}


/*
#include <iostream>
#include <vector>
#include <CL/opencl.hpp>
#include <opencv2/core.hpp>


int main() 
{
    cv::Mat matrix = cv::Mat::zeros(3, 3, CV_32F);

    // Set some values in the matrix
    matrix.at<float>(0, 0) = 1.0f;
    matrix.at<float>(1, 1) = 2.0f;
    matrix.at<float>(2, 2) = 3.0f;

    // Print the matrix
    std::cout << "Matrix created with OpenCV:" << std::endl;
    std::cout << matrix << std::endl;

    // Perform a basic operation (transpose the matrix)
    cv::Mat transposed = matrix.t();

    // Print the transposed matrix
    std::cout << "Transposed matrix:" << std::endl;
    std::cout << transposed << std::endl;


    int a = 1 + 2;
    std::cout << "Hello, world!" << std::endl; 
    //return 0;


    try 
    {
        // Get all OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) 
        {
            std::cout << "No OpenCL platforms found." << std::endl;
            return 1;
        }

        // Iterate through platforms
        for (const auto & platform : platforms) 
        {
            std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

            // Get all devices for the platform
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            if (devices.empty()) 
            {
                std::cout << "  No devices found for this platform." << std::endl;
            } 
            else 
            {
                // Print device names
                for (const auto & device : devices) 
                {
                    std::cout << "  Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
                }
            }
        }
    }
    catch (cl::Error & err) 
    {
        std::cerr << "OpenCL Error: " << err.what() << " (" << err.err() << ")" << std::endl;
        return 1;
    }


    

    return 0;
}
*/

/*
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

int main() 
{
    constexpr size_t N = 1024;
    std::vector<float> A(N, 1.0f);  // Initialize all elements to 1.0
    std::vector<float> B(N, 2.0f);  // Initialize all elements to 2.0
    std::vector<float> C(N, 0.0f);  // Result vector

    sycl::queue q(sycl::gpu_selector{});  // Use a GPU device

    // Create buffers for SYCL
    {
        sycl::buffer<float, 1> bufA(A.data(), sycl::range<1>(N));
        sycl::buffer<float, 1> bufB(B.data(), sycl::range<1>(N));
        sycl::buffer<float, 1> bufC(C.data(), sycl::range<1>(N));

        // Submit the kernel for execution
        q.submit([&](sycl::handler& h) 
        {
            auto accA = bufA.get_access<sycl::access::mode::read>(h);
            auto accB = bufB.get_access<sycl::access::mode::read>(h);
            auto accC = bufC.get_access<sycl::access::mode::write>(h);

            h.parallel_for
            (
                sycl::range<1>(N), 
                [=](sycl::id<1> i) 
                {
                    accC[i] = accA[i] + accB[i];  // Vector addition
                }
            );
        });
    } // Buffers go out of scope, ensuring synchronization

    // Print a few results
    std::cout << "Result: ";
    for (size_t i = 0; i < 10; ++i) 
    {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
*/