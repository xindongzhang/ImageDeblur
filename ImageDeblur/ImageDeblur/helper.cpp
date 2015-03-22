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
		return;
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
		return;
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
		return;
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

void Helper::MeshGrid(const int Xsize,
	                  const int Ysize,
					  cv::Mat&  XMat,
					  cv::Mat&  YMat)
{
	if (!XMat.empty() || !YMat.empty()){
		std::cout<< "you should input empty Xmat and Ymat"<<std::endl;
		return;
	}

	XMat = cv::Mat::zeros(Ysize, Xsize, CV_64F);
	YMat = cv::Mat::zeros(Ysize, Xsize, CV_64F);
	for (int i = 0; i < Ysize; ++i)
	{
		for (int j = 0; j < Xsize; ++j)
		{
			XMat.at<double>(i,j) = j+1;
			YMat.at<double>(i,j) = i+1;
		}
	}
}

void Helper::warpProjective2(const cv::Mat img,
	                         cv::Mat A,
	                         cv::Mat& result)
{
	if (A.rows > 2){
		A = A(cv::Range(0,2), cv::Range(0,A.cols));
	}
	cv::Mat x,y;
	Helper::MeshGrid(img.cols, img.rows, x, y);
	cv::Mat homogeneousCoords = cv::Mat::zeros(3, x.rows*x.cols, CV_64F);
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < x.rows*x.cols; ++j)
		{
			if (i == 0){
				homogeneousCoords.at<double>(i,j) = x.at<double>( (j)%x.cols, std::floor(double(j)/x.rows));
			} 
			if (i == 1){
				homogeneousCoords.at<double>(i,j) = y.at<double>( (j)%y.cols, std::floor(double(j)/y.rows));
			}
			if (i == 2){
				homogeneousCoords.at<double>(i,j) = 1.0;
			}
		}
	}
	/*------------------------*/
	cv::Mat warpedCoords = A * homogeneousCoords;
	cv::Mat xprime = cv::Mat::zeros(img.size(), CV_64F);// = warpedCoords(cv::Range(0,1),cv::Range(0,warpedCoords.cols));
	cv::Mat yprime = cv::Mat::zeros(img.size(), CV_64F);// = warpedCoords(cv::Range(1,2),cv::Range(0,warpedCoords.cols));

	for (int i = 0; i < warpedCoords.cols; ++i)
	{
		int R = (i)%img.cols;
		int C = std::floor(double(i)/img.rows);
		xprime.at<double>(R,C) = warpedCoords.at<double>(0, i);
		yprime.at<double>(R,C) = warpedCoords.at<double>(1, i);
	}

	xprime.convertTo(xprime, CV_32F);
	yprime.convertTo(yprime, CV_32F);
	/*------------------------*/
	cv::remap(img, result, yprime, xprime, CV_INTER_LINEAR,0,cv::Scalar(0,0,0));
	cv::flip(result, result,2);
	result = result.t();
}

void Helper::warpimage(const cv::Mat img, 
	                   const cv::Mat M,
	                   cv::Mat& warped)
{
	Helper::warpProjective2(img, M, warped);
}

void Helper::adjust_psf_center(const cv::Mat kernel,
	                           cv::Mat& adjusted)
{
	cv::Mat X;
	cv::Mat Y;
	Helper::MeshGrid(kernel.cols, kernel.rows, X, Y);
	std::vector<double> xc1, yc1, xc2, yc2;
	Helper::Sum(kernel.mul(X), xc1, ALL_SUM);
	Helper::Sum(kernel.mul(Y), yc1, ALL_SUM);
	xc2.push_back((kernel.cols+1)/ 2.0) ;
	yc2.push_back((kernel.rows+1)/ 2.0) ;
	double xshift = cvRound(xc2[0] - xc1[0]);
	double yshift = cvRound(yc2[0] - yc1[0]);
	cv::Mat M = cv::Mat::zeros(2,3,CV_64F);
	M.at<double>(0,0) = 1; 
	M.at<double>(0,2) = -xshift;
	M.at<double>(1,1) = 1; 
	M.at<double>(1,2) = -yshift; 
	Helper::warpimage(kernel, M, adjusted);
}

void Helper::estimate_psf(const cv::Mat blurred_x,
	                      const cv::Mat blurred_y,
	                      const cv::Mat latent_x,
	                      const cv::Mat latent_y,
	                      const double weight,
	                      const int psf_size,
	                      cv::Mat& psf)
{
	/*assume their size are equal*/
	cv::Mat latent_xf;
	cv::Mat latent_xf_RI[] = {cv::Mat_<float>(latent_x), 
		                      cv::Mat::zeros(latent_x.size(), CV_32F)};
	/*---------------------------*/
	cv::Mat latent_yf;
	cv::Mat latent_yf_RI[] = {cv::Mat_<float>(latent_y), 
		                      cv::Mat::zeros(latent_y.size(), CV_32F)};
	/*---------------------------*/
	cv::Mat blurred_xf;
	cv::Mat blurred_xf_RI[] = {cv::Mat_<float>(blurred_x), 
		                       cv::Mat::zeros(blurred_x.size(), CV_32F)};
	/*---------------------------*/
	cv::Mat blurred_yf;
	cv::Mat blurred_yf_RI[] = {cv::Mat_<float>(blurred_y), 
		                       cv::Mat::zeros(blurred_y.size(), CV_32F)};
	/*---------------------------*/
	cv::dft(latent_x, latent_xf, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(latent_y, latent_yf, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(blurred_x, blurred_xf, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(blurred_y, blurred_yf, cv::DFT_COMPLEX_OUTPUT);
	/*-----------split-----------*/
	cv::split(latent_xf, latent_xf_RI);
	cv::split(latent_yf, latent_yf_RI);
	cv::split(blurred_xf, blurred_xf_RI);
	cv::split(blurred_yf, blurred_yf_RI);
	/*----------------------------*/
	cv::Mat latent_xf_RI_Conj[] = {cv::Mat_<float>(latent_x), 
		                           cv::Mat::zeros(latent_x.size(), CV_32F)};
	cv::Mat latent_xf_Conj;
	cv::merge(latent_xf_RI_Conj, 2, latent_xf_Conj);

	cv::Mat latent_yf_RI_Conj[] = {cv::Mat_<float>(latent_y), 
		                           cv::Mat::zeros(latent_y.size(), CV_32F)};
	cv::Mat latent_yf_Conj;
	cv::merge(latent_yf_RI_Conj, 2, latent_yf_Conj);
	/*----------------------------*/
	latent_xf_RI_Conj[0] = latent_xf_RI[0];
	latent_xf_RI_Conj[1] = -latent_xf_RI[1];
	latent_yf_RI_Conj[0] = latent_yf_RI[0];
	latent_yf_RI_Conj[1] = -latent_yf_RI[1];
	/*----------------------------*/
	cv::Mat b_f = latent_xf_Conj.mul(blurred_xf) + 
		          latent_yf_Conj.mul(blurred_yf);
}


void Helper::psf2otf(const cv::Mat psf, 
	                 const cv::Size size,
	                 cv::Mat& otf)
{
	/*assume that all the elements are non-zero*/
	/*should padding*/
	int R = (size.height - psf.cols)/2;
	int C = (size.width - psf.rows)/2;
	cv::Mat t_psf = cv::Mat::zeros(size, CV_32F);
	psf.copyTo(t_psf(cv::Range(R,t_psf.rows-R),cv::Range(C,t_psf.cols-C)));
	Helper::circshift(t_psf,cv::Size(-size.width/2,-size.height/2),t_psf);
	cv::dft(t_psf, otf, cv::DFT_COMPLEX_OUTPUT);
}

void Helper::otf2psf(const cv::Mat otf,
	                 const cv::Size size,
	                 cv::Mat& psf)
{
	/*assume that all the elements are non-zero*/
	/*without padding*/
	int R = (size.height - psf.cols)/2;
	int C = (size.width - psf.rows)/2;
	cv::Mat t_otf = cv::Mat::zeros(size, CV_32F);
	otf.copyTo(t_otf(cv::Range(R,t_otf.rows-R),cv::Range(C,t_otf.cols-C)));
	cv::dft(t_otf, psf, cv::DFT_INVERSE|cv::DFT_SCALE);
	Helper::circshift(psf, cv::Size(size.width/2,size.height/2),psf);
}

void Helper::circshift(const cv::Mat& src,
	                   const cv::Size size,
	                   cv::Mat& dst)
{
	cv::Mat t_src = src.clone();
	dst = src.clone();
	/*----------?----------*/
	int R = size.height % src.rows;
	int C = size.width  % src.cols;
	int rows = src.rows;
	int cols = src.cols;
	/*---------------------*/
	if (R > 0) {
		for (int i = 0; i < rows; ++i)
		{
		    t_src(cv::Range((i+R)%rows,(i+R)%rows+1), cv::Range(0,cols)).
				copyTo(dst(cv::Range(i,i+1),cv::Range(0,cols)));
		}
	} else {
		for (int i = 0; i < rows; ++i)
		{
		    t_src(cv::Range((i+rows+R)%rows,(i+rows+R)%rows+1), cv::Range(0,cols)).
				copyTo(dst(cv::Range(i,i+1),cv::Range(0,cols)));
		}
	}
	/*--------------------*/
	t_src = dst.t();
	dst   = dst.t();
	if (C > 0) {
		for (int i = 0; i < cols; ++i)
		{
			t_src(cv::Range((i+C)%cols,(i+C)%cols+1), cv::Range(0,rows)).
				copyTo(dst(cv::Range(i,i+1),cv::Range(0,rows)));
		}
	} else {
		for (int i = 0; i < cols; ++i)
		{
			t_src(cv::Range((i+cols+C)%cols,(i+cols+C)%cols+1), cv::Range(0,rows)).
				copyTo(dst(cv::Range(i,i+1),cv::Range(0,rows)));
		}
	}
	/*--------------------*/
	dst = dst.t();
}
