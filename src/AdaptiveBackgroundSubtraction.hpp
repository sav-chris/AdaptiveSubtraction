// Author: Christopher Savini
#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_TARGET_OPENCL_VERSION 300

#include <opencv2/opencv.hpp>
#include <CL/opencl.hpp>
#include <string> 


class AdaptiveBackgroundSubtraction
{
private:
	static std::string readFile(std::string filename);
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Platform platform;
	cl::Device device;
	cl::Program::Sources sources; 
	std::string kernelCode;

	cl::Context context;
	cl::Program program;

	cl::CommandQueue queue;

	cl::Kernel kernelCalcDF;
	cl::Kernel kernelCalcDBDI;
	cl::Kernel kernelCalcPAVG;
	cl::Kernel kernelAdaptiveBackgroundSubtraction;

	std::vector<float> imageBuffer;
	std::vector<float> backgroundBuffer;
	std::vector<float> dIdxBuffer;
	std::vector<float> dIdyBuffer;
	std::vector<float> dBdxBuffer;
	std::vector<float> dBdyBuffer;
	std::vector<float> pBuffer;
	std::vector<float> AVGBuffer;
	std::vector<float> signalBuffer;

	cl::Buffer imageGpuBuffer;
	cl::Buffer backgroundGpuBuffer;
	cl::Buffer dIdxGpuBuffer;
	cl::Buffer dIdyGpuBuffer;
	cl::Buffer dBdxGpuBuffer;
	cl::Buffer dBdyGpuBuffer;
	cl::Buffer pGpuBuffer;
	cl::Buffer AVGGpuBuffer;
	cl::Buffer signalGpuBuffer;

	void initialiseGPU();
	void allocateCalcDBDI(int length);
	void allocateCalcPAVG(int length);
	void allocateAdaptiveBackgroundSubtraction(int length);

	void CalcDBDI(int width, int height);
	void CalcPAVG(int width, int height, int tau);
	void AdaptiveBackgroundSubtractionKernel
	(
		int length,
		int width,
		int height,
		int tau,
		float average,
		cv::Mat & signal
	);

public:
	void CalcDF
	(
		const cv::Mat & image,
		cv::Mat & dIdx,
		cv::Mat & dIdy
	);

	void AdaptiveSubtraction
	(
		const cv::Mat & image,
		const cv::Mat & background,
		int tau,
		cv::Mat & signal
	);

	void ColourAdaptiveSubtraction
	(
		const cv::Mat & image,
		const cv::Mat & background,
		int tau,
		cv::Mat & signal
	);

	AdaptiveBackgroundSubtraction(int length);

	float Grad(cv::Mat & image1, cv::Mat & image2);
};

