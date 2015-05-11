#include "common.h"
#include "ImageDeblur.h"
#include "helper.h"



int main(void)
{
	std::string filename = "./image/roma.png";
	cv::Mat img = cv::imread(filename, 0);
	// ImageDeblur
	ImageDeblur ID(img);
	cv::Mat latent;
	cv::Mat kernel = cv::Mat::zeros(55,55, CV_32F);
	struct Opts opts;
	Helper::setOpts(opts,1,5,1.0,20,55,0,4e-4,4e-4);
	ID.blind_deconv(img, opts, kernel);
	//float SUM = cv::sum(kernel).val[0];
	//kernel = kernel.mul(1.0/SUM * 255 * 255);
	//Helper::printMat(kernel);
	//kernel.convertTo(kernel, CV_8U);
	//cv::imshow("kernel", kernel);
	ID.blind_deblur(img, kernel, 0.001, latent);
	latent.convertTo(latent, CV_8U);
	cv::imshow("blur", img);
	cv::imshow("latent", latent);
	cv::waitKey();
	return 0;
}