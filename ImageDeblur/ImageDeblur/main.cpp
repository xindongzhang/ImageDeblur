#include "common.h"
#include "ImageDeblur.h"
#include "helper.h"



int main(void)
{
	std::string filename = "./image/11.bmp";
	cv::Mat img = cv::imread(filename, 0);
	cv::imshow("img", img);

	// ImageDeblur
	ImageDeblur ID(img);
	cv::Mat latent;
	cv::Mat kernel;
	struct Opts opts;
	Helper::setOpts(opts,1,5,1.0,20,23,0,4e-4,4e-4);
	ID.blind_deconv(img, opts, latent, kernel);
	// downsample
	cv::Mat downsample;
	cv::pyrDown(img, downsample, cv::Size(img.cols*0.5, img.rows*0.5));
	cv::imshow("downsample", downsample);
	cv::waitKey();
	return 0;
}