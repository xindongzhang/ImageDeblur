#include "helper.h"

void Helper::setOpts(struct Opts& opts,
	                double prescale0, int xk_iter0,
	                double gamma_correct0, int k_thresh0,
	                int kernel_size0, int saturation0,
	                double lambda_pixel0, double lambda_grad0)
{
	opts.prescale  = prescale0;
	opts.xk_iter = xk_iter0;
	opts.gamma_correct = gamma_correct0;
	opts.k_thresh = k_thresh0;
	opts.kernel_size = kernel_size0;
	opts.saturation = saturation0;
	opts.lambda_pixel = lambda_pixel0;
	opts.lambda_grad = lambda_grad0;
}

void Helper::init_kernel(const int minsize, cv::Mat& kernel)
{
	if (minsize%2 != 1)
	{
		std::cout<< "the minsize must be odd number!"<< std::endl;
		return;
	}
	kernel = cv::Mat::zeros(minsize, minsize, CV_64F);
	kernel.at<double>((minsize-1)/2, (minsize-1)/2) = 0.5;
	kernel.at<double>((minsize-1)/2, (minsize-1)/2 + 1) = 0.5;
}

void Helper::fixsize(const int nk1,
	                 const int nk2,
					 cv::Mat& kernel)
{
	int k1 = kernel.rows;
	int k2 = kernel.cols;
	/*iterations*/
	while( (k1!=nk1) || (k2!=nk2) )
	{
		/*---------------*/
		if (k1>nk1)
		{
			std::vector<double> s;
			Helper::Sum(kernel, s, 2);
			if ( *s.begin() < *(s.end()-1) ){
				kernel = kernel(cv::Range(1, s.size()), 
					            cv::Range(0, kernel.cols));
			} else {
				kernel = kernel(cv::Range(1, s.size()-1), 
					            cv::Range(0, kernel.cols));
			}
		}
		/*---------------*/
		if (k1<nk1)
		{
			std::vector<double> s;
			Helper::Sum(kernel, s, 2);
			if ( *s.begin() < *(s.end()-1) ){
				cv::Mat tk = cv::Mat::zeros(k1+1, kernel.cols, CV_64F);
				tk(cv::Range(0,k1),cv::Range(0, tk.cols)) = kernel;
				kernel = tk;
			} else {
				cv::Mat tk = cv::Mat::zeros(k1+1, kernel.cols, CV_64F);
				tk(cv::Range(1,k1+1),cv::Range(0, tk.cols)) = kernel;
				kernel = tk;
			} 
		}
		/*-----------------*/
		if (k2>nk2)
		{
			std::vector<double> s;
			Helper::Sum(kernel, s, 1);
			if (*s.begin() < *(s.end()-1)){
				kernel = kernel(cv::Range(0,kernel.rows),cv::Range(1,kernel.cols));
			} else {
				kernel = kernel(cv::Range(0,kernel.rows),cv::Range(0,kernel.cols-1));
			}
		}
		/*----------------*/
		if (k2<nk2)
		{
			std::vector<double> s;
			Helper::Sum(kernel, s, 1);
			if (*s.begin() < *(s.end()-1)) {
				cv::Mat tk = cv::Mat::zeros(kernel.rows, k2+1,CV_64F);
				tk(cv::Range(0,tk.rows),cv::Range(0,k2)) = kernel;
				kernel = tk;
			} else {
				cv::Mat tk = cv::Mat::zeros(kernel.rows, k2+1,CV_64F);
				tk(cv::Range(0,tk.rows),cv::Range(1,k2+1)) = kernel;
				kernel = tk;
			}
		}
		k1 = kernel.rows;
		k2 = kernel.cols;
	}
}


/*it is curious that can not use #if and #elif*/
void Helper::Sum(const cv::Mat src,
	std::vector<double>& result,
	const int flag)
{

	/*check if the result is empty or not*/
	if (!result.empty()){
		result.clear();
	}
	/*row sum*/
	if (flag == ROW_SUM)
	{
		double tmp_sum;
		for (int i = 0; i < src.rows; ++i)
		{
			tmp_sum = 0.0;
			for (int j = 0; j < src.cols; ++j)
			{
				tmp_sum += src.at<double>(i,j);
			}
			result.push_back(tmp_sum);
		}
	}
	/*col sum*/
	if (flag == COL_SUM)
	{
		double tmp_sum;
		for (int j = 0; j < src.cols; ++j)
		{
			tmp_sum = 0.0;
			for (int i = 0; i < src.rows; ++i)
			{
				tmp_sum +=src.at<double>(i,j);
			}
			result.push_back(tmp_sum);
		}
	}
	/*all sum*/
	if (flag == ALL_SUM)
	{
		double tmp_sum = 0.0;
		for (int i = 0; i < src.rows; ++i)
		{
			for (int j = 0; j < src.cols; ++j)
			{
				tmp_sum += src.at<double>(i,j);
			}
		}
		result.push_back(tmp_sum);
	}
}

void Helper::resizeKer(const double ret,
	                   const int k1, 
	                   const int k2, 
	                   cv::Mat& kernel)
{
	/*---------resize---------*/
	cv::resize(kernel, kernel, cv::Size(kernel.cols*ret, kernel.rows*ret));
	kernel = kernel.mul(cv::max(kernel, 0));
	Helper::fixsize(k1,k2,kernel);
	double MAX,MIN;
	cv::minMaxIdx(kernel, &MIN, &MAX);
	if (MAX > 0)
	{
		std::vector<double> allsum;
		Helper::Sum(kernel, allsum, ALL_SUM);
		kernel = kernel.mul(1.0/allsum[0]);
	}
}