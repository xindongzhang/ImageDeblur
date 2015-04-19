#include "common.h"
#include "ImageDeblur.h"
#include "helper.h"



int main(void)
{
	std::string filename = "./image/11.bmp";
	cv::Mat img = cv::imread(filename, 0);
	// ImageDeblur
	ImageDeblur ID(img);
	cv::Mat latent;
	cv::Mat kernel;
	struct Opts opts;
	Helper::setOpts(opts,1,5,1.0,20,55,0,4e-4,4e-4);
	ID.blind_deconv(img, opts, kernel);
	ID.blind_deblur(img, kernel, 0.001, latent);
	cv::imshow("blur", img);
	cv::imshow("latent", latent);
	cv::waitKey();
	return 0;
}