#define BOOST_TEST_MODULE AddNumbersTest
#include <boost/test/included/unit_test.hpp>
#include "AdaptiveBackgroundSubtraction.hpp"

//#define CL_TARGET_OPENCL_VERSION 300

BOOST_AUTO_TEST_CASE(AddNumbersTest) 
{
    BOOST_CHECK_EQUAL(2 + 3, 5);
    BOOST_CHECK_EQUAL(-1 + 1, 0); 
    BOOST_CHECK_EQUAL( 0 + 0, 0); 
    BOOST_CHECK_EQUAL(-5 + -5, -10); 
}

void runTest
(
    std::string imgPath, 
    std::string backPath, 
    std::string IMinusB, 
    std::string newBackgroundPath,
    std::string signalPath,
    std::string newComposite
)
{
    cv::Mat image = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
	cv::Mat background = cv::imread(backPath, cv::IMREAD_UNCHANGED);
    
    cv::imwrite(IMinusB, image - background);
    
    cv::Mat newBackground = cv::imread(newBackgroundPath, cv::IMREAD_UNCHANGED);
    newBackground.convertTo(newBackground, CV_32FC3);
    
	image.convertTo(image, CV_32FC3);
	background.convertTo(background, CV_32FC3);

	int length = image.total() * image.elemSize1();
	AdaptiveBackgroundSubtraction adaptiveBackgroundSubtraction = AdaptiveBackgroundSubtraction(length);

    std::vector<int> taus = {5, 10, 15, 30, 100};
    
	cv::Mat signal = cv::Mat(image.size(), CV_32FC3);
    
    for (int& tau: taus)
    {
	    adaptiveBackgroundSubtraction.ColourAdaptiveSubtraction
	    (
		    image,
		    background,
		    tau,
		    signal
	    );

	    std::string filename = signalPath + "[" + std::to_string(tau) + "].png";
        cv::imwrite(filename, signal);
    
        cv::imwrite( newComposite + "[" + std::to_string(tau) + "].png", newBackground + signal);
    }   
}


BOOST_AUTO_TEST_CASE(AdaptiveBackgroundSubtraction_TestWaterBlocks)
{
    //system("mkdir /results/results_table/");
    
    runTest
    (
        "/app/data/water-blocks/water-blocks.png", 
        "/app/data/water-blocks/water-blocks-background.png", 
        "/app/results/water-blocks/IminusB.lid.png", 
        "/app/data/water-blocks/bloodMarble.png",
        "/app/results/water-blocks/water-blocks.foreground",
        "/app/results/water-blocks/water-blocks.newComposite"
    );

    BOOST_CHECK(true); 
}
