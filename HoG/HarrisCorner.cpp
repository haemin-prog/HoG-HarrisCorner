
#include "HarrisCorner.h"

using namespace std;
using namespace cv;


void HarrisCorner::CalculateIxIy() {

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {

			// Simple Edge Filter 
			//int gyValue = inputImg.at<uchar>(y, x) - inputImg.at<uchar>(y - 1, x);
			//int gxValue = inputImg.at<uchar>(y, x) - inputImg.at<uchar>(y, x - 1);

			// Sobel Edge Filter
			int gyValue = inputImg.at<uchar>(y + 1, x) - inputImg.at<uchar>(y - 1, x);
			int gxValue = inputImg.at<uchar>(y, x + 1) - inputImg.at<uchar>(y, x - 1);

			IyImg.at<uchar>(y, x) = abs(gyValue);
			IxImg.at<uchar>(y, x) = abs(gxValue);

			Iy2Img.at<float>(y, x) = (gyValue)*(gyValue);
			Ix2Img.at<float>(y, x) = (gxValue)*(gxValue);
			IxyImg.at<float>(y, x) = (gxValue)*(gyValue);

		}
	}

	Reshow("IxImg", IxImg, 2);
	Reshow("IyImg", IyImg, 2);

	//Reshow("Ix2Img", Ix2Img, 2);
	//Reshow("Iy2Img", Iy2Img, 2);
	//Reshow("IxyImg", IxyImg, 2);

	imwrite(output_IyImgPath, IyImg);
	imwrite(output_IxImgPath, IxImg);

}

void HarrisCorner::GaussianFiltering() {

	double gau[5][5] =
	{ { 1, 4, 6, 4, 1 },{ 4, 16, 24, 16, 4 },
	{ 6, 24, 36, 24, 6 },{ 4, 16, 24, 16, 4 },{ 1, 4, 6, 4, 1 } };

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			gau[i][j] = gau[i][j] / 256.f;
		}
	}

	for (int y = 2; y < height - 2; y++) {
		for (int x = 2; x < width - 2; x++) {

			double x2Value = 0, y2Value = 0, xyValue = 0;

			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {

					x2Value += (Ix2Img.at<float>(y + i - 2, x + j - 2) * gau[i][j]);
					y2Value += (Iy2Img.at<float>(y + i - 2, x + j - 2) * gau[i][j]);
					xyValue += (IxyImg.at<float>(y + i - 2, x + j - 2) * gau[i][j]);

				}
			}

			gIx2Img.at<float>(y, x) = x2Value;
			gIy2Img.at<float>(y, x) = y2Value;
			gIxyImg.at<float>(y, x) = xyValue;

		}
	}

	Reshow("Ix2Img", gIx2Img, 2);
	Reshow("Iy2Img", gIy2Img, 2);
	Reshow("IxyImg", gIxyImg, 2);

}

void HarrisCorner::FindeResponseR() {

	Mat tempRImg[3], showRImg;
	tempRImg[0] = Mat::zeros(height, width, CV_8U);
	tempRImg[1] = Mat::zeros(height, width, CV_8U);
	tempRImg[2] = Mat::zeros(height, width, CV_8U);

	float alpha = 0.04; // 0.05, 0.06

	float minVal = 987654321, maxVal = -1;

	for (int y = 2; y < height - 2; y++) {
		for (int x = 2; x < width - 2; x++) {

			if (axisAlignedCorner) {
				float det = gIx2Img.at<float>(y, x) * gIy2Img.at<float>(y, x) - alpha * (gIx2Img.at<float>(y, x) + gIy2Img.at<float>(y, x)) *(gIx2Img.at<float>(y, x) + gIy2Img.at<float>(y, x));
			}
			else {
				float det = gIx2Img.at<float>(y, x) * gIy2Img.at<float>(y, x) - gIxyImg.at<float>(y, x) * gIxyImg.at<float>(y, x) - alpha * (gIx2Img.at<float>(y, x) + gIy2Img.at<float>(y, x)) *(gIx2Img.at<float>(y, x) + gIy2Img.at<float>(y, x));
			}


			// Original det 
			float det = gIx2Img.at<float>(y, x) * gIy2Img.at<float>(y, x) - gIxyImg.at<float>(y, x) * gIxyImg.at<float>(y, x) - alpha * (gIx2Img.at<float>(y, x) + gIy2Img.at<float>(y, x)) *(gIx2Img.at<float>(y, x) + gIy2Img.at<float>(y, x));

			resRImg.at<float>(y, x) = det;

			minVal = min(minVal, det);
			maxVal = max(maxVal, det);
			// cout << det << " ";

			double r, g, b;

			HSVtoRGB(det, 80, 60, r, g, b);
			tempRImg[0].at<uchar>(y, x) = r;
			tempRImg[1].at<uchar>(y, x) = g;
			tempRImg[2].at<uchar>(y, x) = b;

			merge(tempRImg, 3, showRImg);


		}
		// cout << "\n";
	}

	cout << minVal << " " << maxVal << "\n";
	// -4.54069e+06 5.26988e+07

	Reshow("resRImg", showRImg, 2);

}

void HarrisCorner::FindHarrisCorner() {

	int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	for (int y = 2; y < height - 2; y++) {
		for (int x = 2; x < width - 2; x++) {

			float rValue = resRImg.at<float>(y, x);

			bool nmsFlag = true;  

			if (rValue > threshold) {

				thrImg.at<uchar>(y, x) = 255;

				for (int i = 0; i < 8; i++) {

					int ny = y + dy[i];
					int nx = x + dx[i];

					float neighborValue = resRImg.at<float>(ny, nx);
					if (neighborValue > rValue) {
						nmsFlag = false;
						break;
					}

				}

				if (nmsFlag == true) {
					cornerLoc.push_back({ y,x });
					cout << y << " " << x << " " << rValue << "\n";
				}

			}
			else {
				thrImg.at<uchar>(y, x) = 0;
			}

		}
	}

	Reshow("Threshold Image", thrImg, 2);

}

void HarrisCorner::ShowCorners() {

	Mat showCnrImg = imread(imgPath);

	for (int i = 0; i < cornerLoc.size(); i++) {

		int y = cornerLoc[i].first;
		int x = cornerLoc[i].second;

		circle(showCnrImg, Point(x, y), 2, Scalar(0, 0, 255), -1);

	}

	Reshow("Corner Image", showCnrImg, 2);

	imwrite("OutputHarrisCorner.jpg", showCnrImg);

}


void HarrisCorner::Reshow(string name, Mat _img, double size) {

	if (showFlag) {
		Mat Re_show;
		resize(_img, Re_show, Size((_img.cols * size), (_img.rows*size)), 0, 0, INTER_LINEAR);

		imshow(name, Re_show);
		waitKey(0);
	}
	else
		return;

}


void HarrisCorner::InitialSetting(string _imgPath, string _output_IxImgPath, string _output_IyImgPath, double _threshold) {

	imgPath = _imgPath;
	output_IxImgPath = _output_IxImgPath;
	output_IyImgPath = _output_IyImgPath;

	inputImg = imread(imgPath, IMREAD_GRAYSCALE);

	width = inputImg.size().width;
	height = inputImg.size().height;

	threshold = _threshold;

	IxImg = Mat::zeros(height, width, CV_8U);
	IyImg = Mat::zeros(height, width, CV_8U);

	Ix2Img = Mat::zeros(height, width, CV_32F);
	Iy2Img = Mat::zeros(height, width, CV_32F);
	IxyImg = Mat::zeros(height, width, CV_32F);

	gIx2Img = Mat::zeros(height, width, CV_32F);
	gIy2Img = Mat::zeros(height, width, CV_32F);
	gIxyImg = Mat::zeros(height, width, CV_32F);

	resRImg = Mat::zeros(height, width, CV_32F);

	thrImg = Mat::zeros(height, width, CV_8U);

	cout << "Width : " << width << "\n";
	cout << "Height : " << height << "\n";
	cout << "PixelNum : " << (width - 1) * (height - 1) << "\n";

	Reshow("inputImg", inputImg, 2);

}

void HarrisCorner::HSVtoRGB(double h, double s, double v, double &r, double &g, double &b) {

	int i;
	float f, p, q, t;

	h = max(0.0, min(360.0, h));
	s = max(0.0, min(100.0, s));
	v = max(0.0, min(100.0, v));

	s /= 100;
	v /= 100;

	if (s == 0) {
		// Achromatic (grey)
		r = g = b = round(v * 255);
		return;
	}
	else
	{
		h /= 60; // sector 0 to 5
		i = floor(h);
		f = h - i; // factorial part of h
		p = v * (1 - s);
		q = v * (1 - s * f);
		t = v * (1 - s * (1 - f));
		switch (i) {
		case 0:
			r = round(255 * v);
			g = round(255 * t);
			b = round(255 * p);
			break;
		case 1:
			r = round(255 * q);
			g = round(255 * v);
			b = round(255 * p);
			break;
		case 2:
			r = round(255 * p);
			g = round(255 * v);
			b = round(255 * t);
			break;
		case 3:
			r = round(255 * p);
			g = round(255 * q);
			b = round(255 * v);
			break;
		case 4:
			r = round(255 * t);
			g = round(255 * p);
			b = round(255 * v);
			break;
		default: // case 5:
			r = round(255 * v);
			g = round(255 * p);
			b = round(255 * q);
		}
	}

}