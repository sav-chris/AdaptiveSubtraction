// Author: Christopher Savini
#include <iostream>
#include <string> 
#include <vector>
#include <sstream>
#include <fstream>
#include <streambuf>
#include <numeric>
#include <functional>

#include "AdaptiveBackgroundSubtraction.hpp"

using namespace cv;
using namespace std;
using namespace cl;

std::string AdaptiveBackgroundSubtraction::readFile(std::string filename)
{
	std::vector<std::string> lines;
	std::ifstream infile(filename);
	std::string line;
	while (std::getline(infile, line))
	{
		lines.push_back(line);
	}

	std::string all;
	for (const auto &piece : lines) all += piece + "\n";

	return all;
}

void AdaptiveBackgroundSubtraction::initialiseGPU()
{
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
	{
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	std::cout << "Platforms: " << platforms.size() << std::endl; 

    /*
	std::cout << "Available platforms: " << std::endl;
	for (cl::Platform plat : platforms)
	{
		std::cout << plat.getInfo<CL_PLATFORM_NAME>() << std::endl;
	}
	std::cout << std::endl;
    */

    platform = platforms[1];
	
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.size() == 0)
	{
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	
	device = devices[0];

	std::cout << "Devices: " << devices.size() << std::endl;

	sources.push_back({ kernelCode.c_str(), kernelCode.length() });

	context = cl::Context({ device });

	program = cl::Program(context, sources);
	if (program.build({ device }) != CL_SUCCESS)
	{
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		exit(1);
	}

	queue = cl::CommandQueue(context, device);
	
	kernelCalcDF = cl::Kernel(program, "CalcDF");
	kernelCalcDBDI = cl::Kernel(program, "CalcDBDI");
	kernelCalcPAVG = cl::Kernel(program, "CalcPAVG");
	kernelAdaptiveBackgroundSubtraction = cl::Kernel(program, "AdaptiveBackgroundSubtraction");
}

void AdaptiveBackgroundSubtraction::allocateCalcDBDI(int length)
{
	imageBuffer = std::vector<float>(length);
	backgroundBuffer = std::vector<float>(length);
	dIdxBuffer = std::vector<float>(length);
	dIdyBuffer = std::vector<float>(length);
	dBdxBuffer = std::vector<float>(length);
	dBdyBuffer = std::vector<float>(length);

	imageGpuBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, length);
	backgroundGpuBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, length);
	dIdxGpuBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, length);
	dIdyGpuBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, length);
	dBdxGpuBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, length);
	dBdyGpuBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, length);
}

void AdaptiveBackgroundSubtraction::CalcDBDI(int width, int height)
{
	auto length = imageBuffer.size();
	cl_int err = 0;
	err |= queue.enqueueWriteBuffer(imageGpuBuffer, CL_TRUE, 0, length, &imageBuffer[0]);
	err |= queue.enqueueWriteBuffer(backgroundGpuBuffer, CL_TRUE, 0, length, &backgroundBuffer[0]);

	err |= kernelCalcDBDI.setArg(0, imageGpuBuffer);
	err |= kernelCalcDBDI.setArg(1, backgroundGpuBuffer);
	err |= kernelCalcDBDI.setArg(2, dIdxGpuBuffer);
	err |= kernelCalcDBDI.setArg(3, dIdyGpuBuffer);
	err |= kernelCalcDBDI.setArg(4, dBdxGpuBuffer);
	err |= kernelCalcDBDI.setArg(5, dBdyGpuBuffer);

	kernelCalcDBDI.setArg(6, sizeof(cl_int), &width);
	kernelCalcDBDI.setArg(7, sizeof(cl_int), &height);

	err |= queue.enqueueNDRangeKernel(kernelCalcDBDI, cl::NullRange, cl::NDRange(width, height));
	err |= queue.finish();

	err |= queue.enqueueReadBuffer(dIdxGpuBuffer, CL_TRUE, 0, length, &dIdxBuffer[0]);
	err |= queue.enqueueReadBuffer(dIdyGpuBuffer, CL_TRUE, 0, length, &dIdyBuffer[0]);
	err |= queue.enqueueReadBuffer(dBdxGpuBuffer, CL_TRUE, 0, length, &dBdxBuffer[0]);
	err |= queue.enqueueReadBuffer(dBdyGpuBuffer, CL_TRUE, 0, length, &dBdyBuffer[0]);	
}

void AdaptiveBackgroundSubtraction::allocateCalcPAVG(int length)
{
	pBuffer = std::vector<float>(length);
	AVGBuffer = std::vector<float>(length);

	pGpuBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, length);
	AVGGpuBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, length);
}

void AdaptiveBackgroundSubtraction::CalcPAVG(int width, int height, int tau)
{
	auto length = imageBuffer.size();
	cl_int err = 0;
	err |= queue.enqueueWriteBuffer(imageGpuBuffer, CL_TRUE, 0, length, &imageBuffer[0]);
	err |= queue.enqueueWriteBuffer(backgroundGpuBuffer, CL_TRUE, 0, length, &backgroundBuffer[0]);
	err |= queue.enqueueWriteBuffer(dIdxGpuBuffer, CL_TRUE, 0, length, &dIdxBuffer[0]);
	err |= queue.enqueueWriteBuffer(dIdyGpuBuffer, CL_TRUE, 0, length, &dIdyBuffer[0]);
	err |= queue.enqueueWriteBuffer(dBdxGpuBuffer, CL_TRUE, 0, length, &dBdxBuffer[0]);
	err |= queue.enqueueWriteBuffer(dBdyGpuBuffer, CL_TRUE, 0, length, &dBdyBuffer[0]);

	err |= kernelCalcPAVG.setArg(0, imageGpuBuffer);
	err |= kernelCalcPAVG.setArg(1, backgroundGpuBuffer);
	err |= kernelCalcPAVG.setArg(2, dIdxGpuBuffer);

	err |= kernelCalcPAVG.setArg(3, dIdyGpuBuffer);
	err |= kernelCalcPAVG.setArg(4, dBdxGpuBuffer);
	err |= kernelCalcPAVG.setArg(5, dBdyGpuBuffer);

	kernelCalcPAVG.setArg(6, sizeof(cl_int), &width);
	kernelCalcPAVG.setArg(7, sizeof(cl_int), &height);

	err |= kernelCalcPAVG.setArg(8, pGpuBuffer);
	err |= kernelCalcPAVG.setArg(9, AVGGpuBuffer);

	kernelCalcPAVG.setArg(10, sizeof(cl_int), &tau);

	err |= queue.enqueueNDRangeKernel(kernelCalcPAVG, cl::NullRange, cl::NDRange(width, height));
	err |= queue.finish();

	err |= queue.enqueueReadBuffer(pGpuBuffer, CL_TRUE, 0, length, &pBuffer[0]);
	err |= queue.enqueueReadBuffer(AVGGpuBuffer, CL_TRUE, 0, length, &AVGBuffer[0]);
}

void AdaptiveBackgroundSubtraction::allocateAdaptiveBackgroundSubtraction(int length )
{
	signalBuffer = std::vector<float>(length);
	signalGpuBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, length);
}

void AdaptiveBackgroundSubtraction::AdaptiveBackgroundSubtractionKernel
(
	int length,
	int width,
	int height,
	int tau,
	float average,
	Mat & signal
)
{
	cl_int err = 0;
	err |= queue.enqueueWriteBuffer(imageGpuBuffer, CL_TRUE, 0, length, &imageBuffer[0]);
	err |= queue.enqueueWriteBuffer(backgroundGpuBuffer, CL_TRUE, 0, length, &backgroundBuffer[0]);
	err |= queue.enqueueWriteBuffer(signalGpuBuffer, CL_TRUE, 0, length, &signalBuffer[0]);
	err |= queue.enqueueWriteBuffer(pGpuBuffer, CL_TRUE, 0, length, &pBuffer[0]);
	err |= queue.enqueueWriteBuffer(AVGGpuBuffer, CL_TRUE, 0, length, &AVGBuffer[0]);

	err |= kernelAdaptiveBackgroundSubtraction.setArg(0, imageGpuBuffer);
	err |= kernelAdaptiveBackgroundSubtraction.setArg(1, backgroundGpuBuffer);
	err |= kernelAdaptiveBackgroundSubtraction.setArg(2, signalGpuBuffer);

	kernelAdaptiveBackgroundSubtraction.setArg(3, sizeof(cl_int), &width);
	kernelAdaptiveBackgroundSubtraction.setArg(4, sizeof(cl_int), &height);
	kernelAdaptiveBackgroundSubtraction.setArg(5, sizeof(cl_int), &average);

	err |= kernelAdaptiveBackgroundSubtraction.setArg(6, pGpuBuffer);
	err |= kernelAdaptiveBackgroundSubtraction.setArg(7, AVGGpuBuffer);

	kernelAdaptiveBackgroundSubtraction.setArg(8, sizeof(cl_int), &tau);

	err |= queue.enqueueNDRangeKernel(kernelAdaptiveBackgroundSubtraction, cl::NullRange, cl::NDRange(width, height));
	err |= queue.finish();

	err |= queue.enqueueReadBuffer(signalGpuBuffer, CL_TRUE, 0, length, &signalBuffer[0]);

	std::memcpy(signal.data, &signalBuffer[0], length);
}

AdaptiveBackgroundSubtraction::AdaptiveBackgroundSubtraction(int length)
{
	sources = cl::Program::Sources();
	kernelCode = readFile("/app/src/kernels.cl");

	this->initialiseGPU();

	this->allocateCalcDBDI(length);

	this->allocateCalcPAVG(length);

	this->allocateAdaptiveBackgroundSubtraction(length);
}

void AdaptiveBackgroundSubtraction::ColourAdaptiveSubtraction
(
	const cv::Mat & image,
	const cv::Mat & background,
	int tau,
	cv::Mat & signal
)
{
	CV_Assert(image.dims == 2 && image.depth() == CV_32F); 
	CV_Assert(background.dims == 2 && background.depth() == CV_32F);
	CV_Assert(image.channels() == background.channels());
	CV_Assert((image.total() * image.elemSize1()) == imageBuffer.size());
	CV_Assert((background.total() * background.elemSize1()) == imageBuffer.size());

	int channelCount = image.channels();

	std::vector<Mat> bgrImage; 
	split(image, bgrImage);

	std::vector<Mat> bgrbackground; 
	split(background, bgrbackground);

	auto length = image.total() * image.elemSize1();

	std::vector<Mat> bgrSignal;

	for (int i = 0; i < channelCount; i++)
	{
		Mat sig = Mat(image.size(), CV_32FC1);
		this->AdaptiveSubtraction(bgrImage[i], bgrbackground[i], tau, sig);
		bgrSignal.push_back(sig);
	}

	cv::merge(bgrSignal, signal);

}

void AdaptiveBackgroundSubtraction::CalcDF
(
	const cv::Mat & image,
	cv::Mat & dIdx,
	cv::Mat & dIdy
)
{
	CV_Assert(image.dims == 2 && image.depth() == CV_32F && image.channels() == 1);
	CV_Assert((image.total() * image.elemSize1()) == imageBuffer.size());

	int width = image.size().width;
	int height = image.size().height;
	auto length = imageBuffer.size();

	std::memcpy(&imageBuffer[0], image.data, length);

	cl_int err = 0;
	err |= queue.enqueueWriteBuffer(imageGpuBuffer, CL_TRUE, 0, length, &imageBuffer[0]);

	err |= kernelCalcDF.setArg(0, imageGpuBuffer);
	err |= kernelCalcDF.setArg(1, dIdxGpuBuffer);
	err |= kernelCalcDF.setArg(2, dIdyGpuBuffer);

	kernelCalcDF.setArg(3, sizeof(cl_int), &width);
	kernelCalcDF.setArg(4, sizeof(cl_int), &height);

	err |= queue.enqueueNDRangeKernel(kernelCalcDF, cl::NullRange, cl::NDRange(width, height));
	err |= queue.finish();

	err |= queue.enqueueReadBuffer(dIdxGpuBuffer, CL_TRUE, 0, length, &dIdxBuffer[0]);
	err |= queue.enqueueReadBuffer(dIdyGpuBuffer, CL_TRUE, 0, length, &dIdyBuffer[0]);

	std::memcpy(dIdx.data, &dIdxBuffer[0], length);
	std::memcpy(dIdy.data, &dIdyBuffer[0], length);
}

void AdaptiveBackgroundSubtraction::AdaptiveSubtraction
(
	const cv::Mat & image,
	const cv::Mat & background,
	int tau,
	cv::Mat & signal
)
{
	CV_Assert(image.dims == 2 && image.depth() == CV_32F && image.channels() == 1);
	CV_Assert(background.dims == 2 && background.depth() == CV_32F && background.channels() == 1);
	CV_Assert(image.size() == background.size());
	CV_Assert((image.total() * image.elemSize1()) == imageBuffer.size());

	auto length = image.total() * image.elemSize1();
	int width = image.size().width;
	int height = image.size().height;

	std::memcpy(&imageBuffer[0], image.data, length);
	std::memcpy(&backgroundBuffer[0], background.data, length);

	CalcDBDI(width, height);

	CalcPAVG(width, height, tau );

	auto sumdBdI =
		std::inner_product(std::begin(dIdxBuffer), std::end(dIdxBuffer), std::begin(dBdxBuffer), 0.0) +
		std::inner_product(std::begin(dIdyBuffer), std::end(dIdyBuffer), std::begin(dBdyBuffer), 0.0);

	auto sumdBdB =
		std::inner_product(std::begin(dBdxBuffer), std::end(dBdxBuffer), std::begin(dBdxBuffer), 0.0) +
		std::inner_product(std::begin(dBdyBuffer), std::end(dBdyBuffer), std::begin(dBdyBuffer), 0.0);

	auto p = sumdBdI / sumdBdB;

	std::vector<float> diff = std::vector<float>(imageBuffer.size());

	std::vector<float> pB = std::vector<float>(backgroundBuffer.size());

	std::transform
	(
		backgroundBuffer.begin(),
		backgroundBuffer.end(),
		pB.begin(),
		std::bind(std::multiplies<float>(), std::placeholders::_1, p)
	);

	std::transform
	(
		imageBuffer.begin(),
		imageBuffer.end(),
		pB.begin(),
		diff.begin(),
		std::minus<float>()
	);

	auto average = std::accumulate(diff.begin(), diff.end(), 0.0) / diff.size();

	AdaptiveBackgroundSubtractionKernel
	(
		length,
		width,
		height,
		tau,
		average,
		signal
	);
}

float AdaptiveBackgroundSubtraction::Grad(cv::Mat & image1, cv::Mat & image2)
{
	Mat dFdx = image1.clone();
	Mat dFdy = image1.clone();
	Mat dIdx = image1.clone();
	Mat dIdy = image1.clone();

	this->CalcDF(image1, dFdx, dFdy);
	this->CalcDF(image2, dIdx, dIdy);

	Mat diffxMat = image1.clone();
	Mat diffyMat = image1.clone();
	cv::subtract(dFdx, dIdx, diffxMat);
	cv::subtract(dFdy, dIdy, diffyMat);

	diffxMat = diffxMat.mul(diffxMat);
	diffyMat = diffyMat.mul(diffyMat);

	Mat squaredMat = diffxMat + diffyMat;

	auto totalMat = cv::sum(squaredMat);

	auto mean = cv::mean(squaredMat);

	return mean[0];
}



