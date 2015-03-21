#ifndef IMAGEDEBLUR_H
#define IMAGEDEBLUR_H

#include "common.h"
#include "helper.h"
class ImageDeblur
{
private:
	cv::Mat img;
	cv::Mat kernel;
	struct Opts opts;
public:
	//default constructor
	ImageDeblur(cv::Mat img0)
	{
		// initialize the image
		if (img0.type() != CV_8U)
		{
			cv::cvtColor(img0, img, cv::COLOR_BGR2GRAY);
		} else {
			this->img = img0.clone();
		}
		// seting the image to double type
		img.convertTo(this->img, CV_64F);
		// set opts
		Helper::setOpts(this->opts,1,5,1.0,20,23,0,4e-4,4e-4);
		std::cout<< opts.kernel_size<< std::endl;
		/*initialize the kernel as double*/
		this->kernel = cv::Mat::zeros(opts.kernel_size, opts.kernel_size, CV_32F);
	}
	//default de-constructor
	~ImageDeblur(){}
	// get method
	cv::Mat getImg()
	{
		if (img.empty())
		{
			std::cout<< "the image is empty"<< std::endl;
		} else {
			return this->img;
		}
	}

	cv::Mat getKernel()
	{
		if (kernel.empty())
		{
			std::cout<< "the image is empty"<< std::endl;
		} else {
			return this->kernel;
		}
	}
	/*blind deconvolution*/
	void blind_deconv(const cv::Mat src, 
		              struct Opts opts,
		              cv::Mat& latent_img, 
					  cv::Mat& kernel);
};

#endif