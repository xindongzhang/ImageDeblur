#include "ImageDeblur.h"


void ImageDeblur::blind_deconv(const cv::Mat src,
	                           struct Opts opts,
							   cv::Mat& kernel)
{
	/*initialize the origin blurry image*/
	cv::Mat img;
	if (src.type() != CV_32F)
	{
		src.convertTo(img, CV_32F);
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
	double cret;
	double lambda = 2E-2;
	for (int i = num_scales; i > 0; --i)
	{
		std::cout<< i << std::endl;
		// processing the kernel
		if ( i == num_scales)
		{
			Helper::init_kernel(k1list[i-1], kernel);
			k1 = k1list[i-1];
			k2 = k1;
		} else {
			k1 = k1list[i-1];
			k2 = k1;
			Helper::resizeKer(ret, k1, k2, kernel);
		}
		// estimate gradient
		cret = std::pow(ret, i-1);
		cv::Mat ys;
		cv::resize(src, ys, cv::Size(src.size().height*cret, src.size().width*cret));
		cv::Mat yx, yy;
		cv::Sobel(ys, yx, CV_32F, 1, 0);
		cv::Sobel(ys, yy, CV_32F, 0, 1);
		//------------//
		this->blind_deconv_main(yx, yy, yx, yy, kernel, lambda, opts);

	}
}

void ImageDeblur::blind_deconv_main(const cv::Mat& yx, 
									const cv::Mat& yy,
									cv::Mat& xx,
									cv::Mat& xy, 
									cv::Mat& kernel, 
									double& lambda0, Opts opts0)
{
	double lambda_orig = lambda0;
	double lambda, r, w;
	r = 4;
	cv::Mat latent_x, latent_y;
	for (int i = 0; i < opts0.xk_iter; ++i)
	{
		lambda = lambda_orig;
		r      = std::sqrt(r);
		w      = lambda * std::exp(-std::pow(std::abs(r), 0.8));
		this->L0deblur_adm(yx, yy, kernel, lambda, opts0, latent_x, latent_y);
		Helper::kernel_solver(yx, yy, latent_x, latent_y, kernel);
		for (int ii = 0; ii < kernel.rows; ++ii)
		{
			for (int jj = 0; jj < kernel.cols; ++jj)
			{
				if (kernel.at<float>(ii,jj) < 0) kernel.at<float>(ii,jj) = 0;
			}
		}
		/*------------------*/
		float SUM = cv::sum(kernel).val[0];
		kernel = kernel.mul(1.0/SUM);
	}
	lambda0 = lambda * 0.55;
}

void ImageDeblur::L0deblur_adm(const cv::Mat& blur_x,
	                           const cv::Mat& blur_y, 
							   const cv::Mat& kernel, 
							   double lambda, 
							   Opts opts0,
							   cv::Mat& latent_x, 
							   cv::Mat& latent_y)
{
	/*assume that all the image are gray scale*/
	int betamax = std::pow(2.0, 8);
	cv::Mat KER;
	cv::Mat DenKER;
	Helper::psf2otf(kernel, blur_x.size(), KER);
	cv::mulSpectrums(KER, KER, DenKER, cv::DFT_REAL_OUTPUT, true);
	cv::Mat Xplanes[] = {blur_x, cv::Mat::zeros(blur_x.size(), CV_32F)};
	cv::Mat Yplanes[] = {blur_y, cv::Mat::zeros(blur_y.size(), CV_32F)};
	cv::Mat Sx,Sy;
	cv::merge(Xplanes, 2, Sx);
	cv::merge(Yplanes, 2, Sy);
	cv::Mat Normin_x, Normin_y;
	cv::dft(Sx, Normin_x, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(Sy, Normin_y, cv::DFT_COMPLEX_OUTPUT);
	/*---------------------*/
	double beta = 0.01;
	double kappa = 3;
	/*---------lagrange multiplier-----------*/
	cv::Mat para[] = {cv::Mat::zeros(Sx.size(), CV_32F),
		              cv::Mat::zeros(Sx.size(), CV_32F)};
	cv::Mat adm_para1;
	cv::Mat adm_para2;
	cv::merge(para, 2, adm_para1);
	cv::merge(para, 2, adm_para2);
	cv::Mat Denormin, h, v, t, FSx, FSy, CONJ_KER;
	while (beta < betamax)
	{
		Denormin = DenKER + beta;
		/*---h-v sub problem---*/
		cv::add(Sx, adm_para1.mul(1.0/(2*beta)), h);
		cv::add(Sy, adm_para2.mul(1.0/(2*beta)), v);
		t = (h.mul(h) + v.mul(v)) < lambda/beta;
		t.convertTo(t, CV_32F);
		/*set to zeros*/
		for (int i = 0; i < t.rows; ++i)
		{
			for (int j = 0; j < t.cols; ++j)
			{
				if (t.at<float>(i,j)!=0)
				{
					h.at<float>(i,j) = 0;
					v.at<float>(i,j) = 0;
				}
			}
		}
		/*-----S subproblem-----*/
		cv::Mat tmp[] = {cv::Mat::zeros(Sx.size(), CV_32F),
			             cv::Mat::zeros(Sx.size(), CV_32F)};
		cv::split(KER, tmp);
		tmp[1] = -tmp[1];
		cv::merge(tmp, 2, CONJ_KER);
		/*------------------------*/
		cv::dft( h-adm_para1.mul(1.0/(2*beta)), FSx, cv::DFT_COMPLEX_OUTPUT);
		cv::dft( v-adm_para1.mul(1.0/(2*beta)), FSy, cv::DFT_COMPLEX_OUTPUT);
		cv::add(FSx.mul(beta), CONJ_KER.mul(Normin_x), FSx);
		cv::add(FSy.mul(beta), CONJ_KER.mul(Normin_y), FSy);
		//FSx = FSx.mul(beta) + CONJ_KER.mul(Normin_x);
		//FSy = FSy.mul(beta) + CONJ_KER.mul(Normin_y);
		cv::divide(FSx, Denormin, FSx);
		cv::divide(FSy, Denormin, FSy);
		/*------------------------*/
		cv::idft(FSx, Sx, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		cv::idft(FSy, Sy, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		/*------update the lagrange multiplier--------*/
		cv::Mat tmpSx[] = {Sx, cv::Mat::zeros(Sx.size(), CV_32F)};
		cv::Mat tmpSy[] = {Sy, cv::Mat::zeros(Sy.size(), CV_32F)};
		cv::merge(tmpSx, 2, Sx);
		cv::merge(tmpSy, 2, Sy);
		//adm_para1 = adm_para1 + (Sx-h) * beta;
		//adm_para2 = adm_para2 + (Sy-v) * beta;
		cv::add(adm_para1, (Sx-h)*beta, adm_para1);
		cv::add(adm_para2, (Sy-v)*beta, adm_para2);
		beta = beta * kappa;
	}
	cv::Mat tmpSx[] = {cv::Mat::zeros(Sx.size(), CV_32F),
	                   cv::Mat::zeros(Sx.size(), CV_32F)};
	cv::Mat tmpSy[] = {cv::Mat::zeros(Sy.size(), CV_32F),
		               cv::Mat::zeros(Sy.size(), CV_32F)};
	cv::split(Sy, tmpSy);
	cv::split(Sx, tmpSx);
	latent_x = tmpSx[0].clone();
	latent_y = tmpSy[0].clone();
}



void ImageDeblur::blind_deblur(const cv::Mat& src, 
	                           const cv::Mat& kernel,
							   double lambda, 
							   cv::Mat& dst)
{
	std::vector<cv::Mat> channels;
	if (src.type() == CV_8U){
		channels.push_back(src);
	}
	else{
		cv::split(src, channels);
	}
	for (int i = 0; i < channels.size(); ++i)
	{
		std::cout<< "channels "<< i<< std::endl;
		this->blind_deblur_aniso(channels[i], kernel,lambda, channels[i]);
	}
	/*-----------*/
	if (channels.size() != 1){
		cv::merge(channels, dst);
	}else{
		dst = channels[0].clone();
	}

}

void ImageDeblur::blind_deblur_aniso(const cv::Mat& blur, 
	                                 const cv::Mat& kernel,
									 double lambda,
									 cv::Mat& latent)
{
	double beta = 1.0 / lambda;
	double beta_min = 10;//0.001
	cv::Mat I = blur.clone();
	cv::Mat Nomin1, Denom1, Denom2, Denom;
	this->computeDenominator(I, kernel, Nomin1, Denom1, Denom2);
	cv::Mat dx = cv::Mat::zeros(1,2,CV_32F);
	dx.at<float>(0) = -1;
	dx.at<float>(1) = 1;
	cv::Mat dy = cv::Mat::zeros(2,1,CV_32F);
	dy.at<float>(0) = -1;
	dy.at<float>(1) = 1;
	/*--------------*/
	cv::Mat Ix, Iy;
	cv::Mat Wx, Wy, Wxx;
	cv::Mat signIx, signIy;
	cv::filter2D(I, Ix, CV_32F, dx, cv::Point(0, 0), 0.0, cv::BORDER_REFLECT);
	cv::filter2D(I, Iy, CV_32F, dy, cv::Point(0, 0), 0.0, cv::BORDER_REFLECT);
	double gamma;
	cv::Mat Wx_x;
	cv::Mat Wy_y;
	cv::Mat Fyout;
	cv::Mat fftWxx;
	while (beta>beta_min)
	{
		gamma = 1 / (2*beta);
		cv::add(Denom1, gamma * Denom2, Denom);
		cv::Mat pos = Ix > 0;
		cv::Mat neg = Ix < 0;
		cv::add(pos, neg, signIx);
		signIx.convertTo(signIx, CV_32F);
		signIx = signIx.mul(1.0 / 255);
		pos = Iy > 0;
		neg = Iy < 0;
		cv::add(pos, neg, signIy);
		signIy.convertTo(signIy, CV_32F);
		signIy = signIy.mul(1.0 / 255);
		/*--------------*/
		cv::Mat tmp;
		cv::max(cv::abs(Ix)-beta*lambda, 0, tmp);
		Wx = tmp.mul(signIx);
		cv::max(cv::abs(Iy)-beta*lambda, 0, tmp);
		Wy = tmp.mul(signIy);
		/*--------------*/
		cv::filter2D(Wx, Wx_x, CV_32F, dx, cv::Point(0, 0), 0.0, cv::BORDER_REFLECT);
		cv::filter2D(Wx, Wy_y, CV_32F, dy, cv::Point(0, 0), 0.0, cv::BORDER_REFLECT);
		Wxx = -Wx_x-Wy_y;
		/*--------------*/
		cv::Mat fftWxx_plane[] = {Wxx, cv::Mat::zeros(Wxx.size(), CV_32F)};
		cv::merge(fftWxx_plane, 2, fftWxx);
		cv::dft(fftWxx, fftWxx, cv::DFT_COMPLEX_OUTPUT);
		//cv::Mat plane[] = {Nomin1, cv::Mat::zeros(Nomin1.size(), CV_32F)};
		//cv::merge(plane, 2, Nomin1);
		cv::add(Nomin1, gamma*fftWxx, Fyout);
		//cv::Mat plane1[] = {Denom, cv::Mat::zeros(Denom.size(), CV_32F)};
		//cv::merge(plane1, 2, Denom);
		cv::divide(Fyout, Denom, Fyout);
		cv::idft(Fyout, I, cv::DFT_REAL_OUTPUT|cv::DFT_SCALE);
		/*--------------*/
		cv::filter2D(I, Ix, CV_32F, dx, cv::Point(0, 0), 0.0, cv::BORDER_REFLECT);
		cv::filter2D(I, Iy, CV_32F, dy, cv::Point(0, 0), 0.0, cv::BORDER_REFLECT);
		beta = beta * 0.5;
		std::cout<< beta<< std::endl;
	}
	latent = I.clone();
}

void ImageDeblur::computeDenominator(const cv::Mat& blur, 
	                                 const cv::Mat& kernel,
									 cv::Mat& Nomin1, 
									 cv::Mat& Denom1, 
									 cv::Mat& Denom2)
{
	cv::Size size = blur.size();
	cv::Mat otfk;
	cv::Mat FFTblur;
	cv::Mat t_blur;
	blur.convertTo(t_blur, CV_32F);
	cv::Mat blur_plane[] = {t_blur, cv::Mat::zeros(blur.size(), CV_32F)};
	cv::merge(blur_plane, 2,FFTblur);
	cv::dft(FFTblur, FFTblur, cv::DFT_COMPLEX_OUTPUT);
	/*------------------*/
	Helper::psf2otf(kernel, size, otfk);
	std::vector<cv::Mat> planes;
	cv::split(otfk, planes);
	planes[1] = -planes[1];
	cv::Mat conj_otfk;
	cv::merge(planes, conj_otfk);
	Nomin1 = conj_otfk.mul(FFTblur);
	cv::mulSpectrums(otfk, otfk, Denom1, cv::DFT_REAL_OUTPUT, true);
	/*------------------*/
	cv::Mat dx = cv::Mat::zeros(1,2,CV_32F);
	dx.at<float>(0) = 1.0;
	dx.at<float>(1) = -1.0;
	cv::Mat dy = cv::Mat::zeros(2,1,CV_32F);
	dy.at<float>(0) = 1.0;
	dy.at<float>(1) = -1.0;
	cv::Mat dx_grat;
	cv::Mat dy_grat;
	Helper::psf2otf(dx, size, dx_grat);
	Helper::psf2otf(dy, size, dy_grat);
	cv::mulSpectrums(dx_grat, dx_grat, dx_grat, cv::DFT_REAL_OUTPUT, true);
	cv::mulSpectrums(dy_grat, dy_grat, dy_grat, cv::DFT_REAL_OUTPUT, true);
	cv::add(dx_grat, dy_grat, Denom2);
}