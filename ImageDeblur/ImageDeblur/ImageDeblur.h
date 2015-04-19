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
		// setting the image to double type
		img.convertTo(this->img, CV_32F);
		// set opts
		Helper::setOpts(this->opts,1,5,1.0,20,23,0,4e-4,4e-4);
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
	// get the kernel of the deblurred algo
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
					  cv::Mat& kernel);
	void blind_deblur(const cv::Mat& src,
		              const cv::Mat& kernel,
					  double lambda,
					  cv::Mat& dst);
private:
	void blind_deconv_main(const cv::Mat& yx, 
		                   const cv::Mat& yy,
		                   cv::Mat& xx,
		                   cv::Mat& xy, 
		                   cv::Mat& kernel, 
		                   double& lambda0, Opts opts0);
	void L0deblur_adm(const cv::Mat& blur_x,
		              const cv::Mat& blur_y,
					  const cv::Mat& kernel,
					  double lambda, Opts opts0,
					  cv::Mat& latent_x,
					  cv::Mat& latent_y);
	void blind_deblur_aniso(const cv::Mat& blur,
		                    const cv::Mat& kernel,
							double lambda,
							cv::Mat& latent);
	void computeDenominator(const cv::Mat& blur,
		                    const cv::Mat& kernel,
							cv::Mat& Nomin1,
							cv::Mat& Denom1,
							cv::Mat& Denom2);
};
#endif