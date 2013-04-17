#include <cstdio>
#include <cuda.h>
#include <opencv2/opencv.hpp>

#include "simple_arrays.h"
#include "dil.h"

int main(int argc, char** argv) {
	using namespace cv;

	cv::Mat img = cv::imread("blob.tif", CV_LOAD_IMAGE_GRAYSCALE);
	cv::resize(img, img, cv::Size(), 10, 5);
	cv::Mat img_se = cv::imread("blob1.tif", CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img, CV_32FC1);
	img_se.convertTo(img_se, CV_32FC1);

	Image32F f;
	f.nd = 2;
	f.dims = new int[2];
	f.dims[0] = img.rows;
	f.dims[1] = img.cols;
	f.size = img.rows * img.cols;
	printf("IMG Size: %d %d\n", img.rows, img.cols);
	f.raster = (char*) img.datastart;

	Image32F se;
	se.nd = 2;
	se.dims = new int[2];
	se.dims[0] = img_se.rows;
	se.dims[1] = img_se.cols;
	printf("SE Size: %d %d\n", img_se.rows, img_se.cols);
	se.size = img_se.rows * img_se.cols;
	se.raster = (char*) img_se.datastart;

	Image32F* g = cudaBinaryDilation(f, se);

	Mat img_g = img.clone();

	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			img_g.at<float>(x,y) = ((float*)(g->raster))[x * img.cols + y];
		}
	}

	//cv::imshow("-1", img);
	cv::imshow("0", img_g);
	cv::imshow("1", img_se);
	cv::waitKey();
	delete g;
	// "f" will be destroyed, than "img". Let's prevent double free ;)
	// same for "se"
	f.raster = NULL;
	se.raster = NULL;
	return 0;
}
