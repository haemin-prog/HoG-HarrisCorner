/*
마감시한: 2020 / 10 / 12 자정
- 보고서는 MS Word 형식으로 작성하여 업로드.소스코드, 결과, 분석 포함.
- 프로그래밍 언어는 제한이 없으나, 기존 영상처리 / 컴퓨터비전 라이브러리의 함수를 이용할 수 없고 반드시 저수준에서 코딩해야 함
*/

#include "stdafx.h"
#include "HOG.h"
#include "HarrisCorner.h"

using namespace std;
using namespace cv;

int main() {

	// 1. HOG(Histogram of oriented gradients) 특징 추출 (Simplified Version) 
	HOG hog;
	hog.InitialSetting("car3.png","MagnitudeImg.jpg", "GradientImg.jpg", "Result.jpg");
	hog.SelectBoundingBox();
	hog.CalculateGradient();
	hog.DrawHistoVector();

	// 2. Harris Corner 검출
	HarrisCorner hc;
	hc.InitialSetting("test.png", "IxImg.jpg", "IyImg.jpg", 5000000);
	hc.CalculateIxIy();
	hc.GaussianFiltering();
	hc.FindeResponseR();
	hc.FindHarrisCorner();
	hc.ShowCorners();

	return 0;
}

///// Global variables
//Mat src, src_gray;
//int thresh = 200;
//int max_thresh = 255;
//
//char* source_window = "test.png";
//char* corners_window = "Corners detected";
//
///// Function header
//void cornerHarris_demo(int, void*);
//
///** @function main */
//int main(int argc, char** argv)
//{
//	/// Load source image and convert it to gray
//	src = imread("test.png", 1);
//	cvtColor(src, src_gray, COLOR_BGR2GRAY);
//
//	/// Create a window and a trackbar
//	namedWindow(source_window, WINDOW_AUTOSIZE);
//	createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
//	imshow(source_window, src);
//
//	cornerHarris_demo(0, 0);
//
//	waitKey(0);
//	return(0);
//}
//
///** @function cornerHarris_demo */
//void cornerHarris_demo(int, void*)
//{
//
//	Mat dst, dst_norm, dst_norm_scaled;
//	dst = Mat::zeros(src.size(), CV_32FC1);
//
//	/// Detector parameters
//	int blockSize = 2;
//	int apertureSize = 3;
//	double k = 0.04;
//
//	/// Detecting corners
//	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
//
//	/// Normalizing
//	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
//	convertScaleAbs(dst_norm, dst_norm_scaled);
//
//	/// Drawing a circle around corners
//	for (int j = 0; j < dst_norm.rows; j++)
//	{
//		for (int i = 0; i < dst_norm.cols; i++)
//		{
//			if ((int)dst_norm.at<float>(j, i) > thresh)
//			{
//				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
//			}
//		}
//	}
//	/// Showing the result
//	namedWindow(corners_window, WINDOW_AUTOSIZE);
//	imshow(corners_window, dst_norm_scaled);
//}