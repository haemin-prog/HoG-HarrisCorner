
#include "HOG.h"

using namespace std;
using namespace cv;

vector<pair<int, int>> tempBoxCoord;

void CallBackFunc(int event, int x, int y, int flag, void* userdata) {

	if (event == EVENT_LBUTTONDOWN) {
		tempBoxCoord.push_back({ y,x });
		cout << "Y Coord : " << y << ",   X Coord : " << x << "\n";
	}

}

void HOG::SelectBoundingBox() {

	cout << "***********************" << endl;
	cout << "Select Bounding Box" << endl << endl;

	Mat tempBoundBoxImg = imread(imgPath);

	imshow("SELECT BOUNDING BOX", tempBoundBoxImg);
	setMouseCallback("SELECT BOUNDING BOX", CallBackFunc, NULL);

	waitKey(0);

	int leftTopX = 987654321, leftTopY = 987654321, rightBottomX = -1, rightBottomY = -1;

	for (int i = 0; i < tempBoxCoord.size(); i++) {

		int y = tempBoxCoord[i].first;
		int x = tempBoxCoord[i].second;

		leftTopX = min(leftTopX, x);
		leftTopY = min(leftTopY, y);

		rightBottomX = max(rightBottomX, x);
		rightBottomY = max(rightBottomY, y);

		circle(tempBoundBoxImg, Point(x, y), 2, Scalar(0, 0, 255), -1);

	}

	rectangle(tempBoundBoxImg, Point(leftTopX, leftTopY), Point(rightBottomX, rightBottomY), Scalar(255, 0, 0), 2);

	imshow("SELECT BOUNDING BOX", tempBoundBoxImg);
	waitKey(0);

	leftTopCorner.first = leftTopY;
	leftTopCorner.second = leftTopX;

	rightBottomCorner.first = rightBottomY;
	rightBottomCorner.second = rightBottomX;

	boundWidth = rightBottomX - leftTopX;
	boundHeight = rightBottomY - leftTopY;

	Rect rect(leftTopX, leftTopY, boundWidth, boundHeight);

	boundBoxImg = inputImg(rect);
	drawImg = tempBoundBoxImg(rect);

	cout << endl;
	cout << "***********************" << endl;
	cout << "Bounding Box Image Information" << endl << endl;

	cout << "Width : " << boundWidth << "\n";
	cout << "Height : " << boundHeight << "\n";

	cout << endl;

	Reshow("BOUNDING BOX Image", boundBoxImg, 1);

}

void HOG::InitialSetting(string _imgPath, string _output_MagnitudePath, string _output_directionPath, string _output_resultPath) {

	imgPath = _imgPath;
	output_MagnitudePath = _output_MagnitudePath;
	output_directionPath = _output_directionPath;
	output_resultPath = _output_resultPath;

	inputImg = imread(imgPath, IMREAD_GRAYSCALE);

	oriWidth = inputImg.size().width;
	oriHeight = inputImg.size().height;

	cout << "***********************" << endl;
	cout << "Input Image Information" << endl << endl;

	cout << "Width : " << oriWidth << "\n";
	cout << "Height : " << oriHeight << "\n";

	cout << endl;

}


void HOG::CalculateGradient() {

	cout << "***********************" << endl;
	cout << "Calculate Gradient : Magnitude & Direction" << endl << endl;

	magnitudeImg = Mat::zeros(boundHeight, boundWidth, CV_8U);
	directionImg = Mat::zeros(boundHeight, boundWidth, CV_8U);

	Mat gx = Mat::zeros(boundHeight, boundWidth, CV_8U);
	Mat gy = Mat::zeros(boundHeight, boundWidth, CV_8U);

	Mat dir[3];
	dir[0] = Mat::Mat::zeros(boundHeight, boundWidth, CV_8U);
	dir[1] = Mat::Mat::zeros(boundHeight, boundWidth, CV_8U);
	dir[2] = Mat::Mat::zeros(boundHeight, boundWidth, CV_8U);

	double minValue = 987654321, maxValue = -1;

	for (int y = 1; y < boundHeight; y++) {
		for (int x = 1; x < boundWidth; x++) {


			int gyValue = boundBoxImg.at<uchar>(y, x) - boundBoxImg.at<uchar>(y - 1, x);
			int gxValue = boundBoxImg.at<uchar>(y, x) - boundBoxImg.at<uchar>(y, x - 1);

			double magnitude = sqrt(gxValue * gxValue + gyValue * gyValue);

			magnitudeImg.at<uchar>(y, x) = (int)magnitude;

			double direction = atan2(gyValue, gxValue);
			direction = direction * 180 / 3.141592;

			minValue = min(minValue, direction);
			maxValue = max(maxValue, direction);

			SaveHistoVector(direction);

			double r, g, b;

			HSVtoRGB(direction, 80, 60, r, g, b);

			dir[0].at<uchar>(y, x) = r;
			dir[1].at<uchar>(y, x) = g;
			dir[2].at<uchar>(y, x) = b;

			merge(dir, 3, directionImg);

		}

	}

	// Normalize 
	for (int i = 0; i < 12; i++) {
		normalVector[i] = (double)histoVector[i] / ((boundWidth - 1)*(boundHeight - 1));
	}

	if (showFlag) {

		cout << "***********************" << endl;
		cout << "Histogram Vector" << endl;
		cout << " -15 ~ 345, 30 deg units" << endl << endl;

		int sum = 0;
		for (int i = 0; i < 12; i++) {
			sum += histoVector[i];
			cout << histoVector[i] << " ";
		}
		cout << "\n\n";

		cout << "***********************" << endl;
		cout << "Normalized Histogram Vector" << endl;
		cout << " -15 ~ 345, 30 deg units" << endl << endl;

		for (int i = 0; i < 12; i++) {
			cout << normalVector[i] << " ";
		}
		cout << "\n\n";

		Reshow("magnitudeImg", magnitudeImg, 2);
		Reshow("directionImg", directionImg, 2);

	}

	cout << "***********************" << endl;
	cout << "Save Magnitude Image & Direction Image" << endl << endl;

	imwrite(output_MagnitudePath, magnitudeImg);
	imwrite(output_directionPath, directionImg);

}

void HOG::SaveHistoVector(double val) {

	val = (int)val;

	if (val <= -15) val += 360;

	if (val > -15 && val <= 15) histoVector[0]++;
	else if (val > 15 && val <= 45) histoVector[1]++;
	else if (val > 45 && val <= 75) histoVector[2]++;
	else if (val > 75 && val <= 105) histoVector[3]++;
	else if (val > 105 && val <= 135) histoVector[4]++;
	else if (val > 135 && val <= 165) histoVector[5]++;
	else if (val > 165 && val <= 195) histoVector[6]++;
	else if (val > 195 && val <= 225) histoVector[7]++;
	else if (val > 225 && val <= 255) histoVector[8]++;
	else if (val > 255 && val <= 285) histoVector[9]++;
	else if (val > 285 && val <= 315) histoVector[10]++;
	else if (val > 315 && val <= 345) histoVector[11]++;

}

void HOG::DrawHistoVector() {

	cout << "***********************" << endl;
	cout << "Draw Gradient Strength : Normalized Histogram Vector" << endl << endl;

	int centerX = boundWidth / 2;
	int centerY = boundHeight / 2;

	float radRangeForOneBin = M_PI / 12;

	// normalVector »ç¿ë

	for (int bin = 0; bin < 12; bin++) {

		float gradientStr = normalVector[bin];

		if (gradientStr == 0) continue;

		float currRad = bin * radRangeForOneBin;// +radRangeForOneBin / 2;

		float dirVecX = cos(currRad);
		float dirVecY = sin(currRad);
		float maxVecLen = min(boundWidth / 2, boundHeight / 2);
		float scale = 3;

		float x1 = centerX -dirVecX * gradientStr * maxVecLen * scale;
		float y1 = centerY -dirVecY * gradientStr * maxVecLen * scale;
		float x2 = centerX + dirVecX * gradientStr * maxVecLen * scale;
		float y2 = centerY + dirVecY * gradientStr * maxVecLen * scale;

		line(drawImg, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 1);

	}

	Reshow("Simplified HOG Result", drawImg, 2);

	cout << "***********************" << endl;
	cout << "Save Gradient Strength Image" << endl << endl;

	imwrite(output_resultPath, drawImg);

}

void HOG::Reshow(string name, Mat _img, double size) {

	if (showFlag) {
		Mat Re_show;
		resize(_img, Re_show, Size((_img.cols * size), (_img.rows*size)), 0, 0, INTER_LINEAR);

		imshow(name, Re_show);
		waitKey(0);
	}
	else
		return;

}


void HOG::HSVtoRGB(double h, double s, double v, double &r, double &g, double &b) {

	int i;
	float f, p, q, t;

	h = max(0.0, min(360.0, h));
	s = max(0.0, min(100.0, s));
	v = max(0.0, min(100.0, v));

	s /= 100;
	v /= 100;

	if (s == 0) {
		r = g = b = round(v * 255);
		return;
	}
	else
	{
		h /= 60; 
		i = floor(h);
		f = h - i; 
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
		default: 
			r = round(255 * v);
			g = round(255 * p);
			b = round(255 * q);
		}
	}

}