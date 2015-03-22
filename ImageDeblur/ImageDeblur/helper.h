#ifndef HELPER_H
#define HELPER_H

#include "common.h"

#define COL_SUM 1
#define ROW_SUM 2
#define ALL_SUM 3


class Helper
{
public:
	static void setOpts(struct Opts& opts,
		                double prescale0 = 1, int xk_iter0 = 5,
		                double gamma_correct0 = 1.0, int k_thresh0 = 20,
						int kernel_size0 = 23, int saturation0 = 0,
						double lambda_pixel0 = 4e-3, double lambda_grad0 = 4e-3);
	static void init_kernel(const int minsize, 
		                    cv::Mat& kernel);
	static void resizeKer(const double ret,
		                  const int k1, 
						  const int k2, 
						  cv::Mat& kernel);
	static void fixsize(const int nk1,
		                const int nk2,
						cv::Mat& kernel);
	static void Sum(const cv::Mat src, 		            
				    std::vector<double>& result,
					const int flag = ROW_SUM);
	static void MeshGrid(const int Xsize,
		                 const int Ysize,
						 cv::Mat&  XMat,
						 cv::Mat&  YMat);
	static void warpProjective2(const cv::Mat img,
		                        const cv::Mat A,
								cv::Mat& result);
	static void warpimage(const cv::Mat img, 
		                  const cv::Mat M,
						  cv::Mat& warped);
	static void adjust_psf_center(const cv::Mat kernel,
		                          cv::Mat& adjusted);
	static void estimate_psf(const cv::Mat blurred_x,
		                     const cv::Mat blurred_y,
							 const cv::Mat latent_x,
							 const cv::Mat latent_y,
							 const double weight,
							 const int psf_size,
							 cv::Mat& psf);
};

#endif