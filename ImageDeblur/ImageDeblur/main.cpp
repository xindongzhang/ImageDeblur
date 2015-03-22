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
	// test helper
	//cv::Mat X,Y;
	//Helper::MeshGrid(3,3,X,Y);
	//cv::Mat Z = X.mul(Y);
	//for (int i = 0; i < 3; ++i)
	//{
	//	for (int j = 0; j < 3; ++j)
	//	{
	//		std::cout<< X.at<double>(i,j)<<" ";
	//		if (j == 2)
	//		{
	//			std::cout<<std::endl;
	//		}
	//	}
	//}
	//cv::Mat result;
	//cv::Mat A = cv::Mat::zeros(2,3,CV_64F);
	//A.at<double>(0,0) = 1.0;
	//A.at<double>(0,2) = 1.0;
	//A.at<double>(1,1) = 1.0;
	//cv::Mat im = cv::Mat::zeros(3,3,CV_64F);
	//im.at<double>(0,0) = 1;
	//im.at<double>(0,1) = 2;
	//im.at<double>(0,2) = 3;
	//im.at<double>(1,0) = 2;
	//im.at<double>(1,1) = 3;
	//im.at<double>(1,2) = 4;
	//im.at<double>(2,0) = 3;
	//im.at<double>(2,1) = 4;
	//im.at<double>(2,2) = 5;
	//Helper::warpProjective2(im, A, result);
	//img.convertTo(img, CV_32F);
	//Helper::estimate_psf(img,img, img,img, 1,1,result);
	//std::cout<< img.rows<<" "<< img.cols<< std::endl;
	cv::Mat psf = cv::Mat::zeros(2,2,CV_32F);
	cv::Mat otf = cv::Mat::zeros(2,2,CV_32F);
	psf.at<float>(0,0) = 1;
	psf.at<float>(0,1) = 2;
	psf.at<float>(1,0) = 3;
	psf.at<float>(1,1) = 4;
	Helper::otf2psf(psf,cv::Size(2,2),otf);
	for (int i = 0; i < otf.rows; ++i)
	{
		for (int j = 0; j < otf.cols; ++j)
		{
			std::cout<< otf.at<float>(i,j)<<" ";
			if (j==otf.cols-1)
			{
				std::cout<< std::endl;
			}
		}
	}
	cv::waitKey();
	return 0;
}