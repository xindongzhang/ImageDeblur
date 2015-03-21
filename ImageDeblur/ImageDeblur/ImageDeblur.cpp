#include "ImageDeblur.h"


void ImageDeblur::blind_deconv(const cv::Mat src,
	                           struct Opts opts,
							   cv::Mat& latent_img,
							   cv::Mat& kernel)
{
	/*initialize the origin blurry image*/
	cv::Mat img;
	if (src.type() != CV_64F)
	{
		src.convertTo(img, CV_64F);
	} else {
		img = src.clone();
	}
	/*gamma correct*/
	if (opts.gamma_correct != 1.0)
	{
		img = img * opts.gamma_correct;
	}

	double ret = std::sqrt(0.5);
	int maxitr = (int)std::max( std::floor(std::log(5.0/opts.kernel_size)/std::log(ret)),0.0);
	int num_scales = maxitr + 1;

	/*initialize the kernel size list*/
	std::vector<int>    k1list;
	std::vector<int>    k2list;
	for (int i = 0; i <= maxitr; ++i)
	{
		int tmp = (int)std::ceil(opts.kernel_size * std::pow(ret, i));
		if (tmp%2 == 0)
			tmp+=1;
		k1list.push_back(tmp);
		k2list.push_back(tmp);
	}

	/*perform iterations*/
	int k1,k2;
	for (int i = num_scales; i > 0; --i)
	{
		if ( i == num_scales)
		{
			Helper::init_kernel(k1list[i-1], kernel);
			k1 = k1list[i-1];
			k2 = k1;
		} else {
			k1 = k1list[i-1];
			k2 = k1;
		}
	}

	/*
	Helper::init_kernel(opts.kernel_size, kernel);
	std::cout<< kernel.at<double>(11,11)<<std::endl;
	img.convertTo(img, CV_8U);
	cv::imshow("gamma correct", img);*/
}