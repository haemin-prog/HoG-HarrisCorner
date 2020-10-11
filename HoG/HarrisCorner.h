#pragma once

#include "stdafx.h"

using namespace std;
using namespace cv;

class HarrisCorner {

public:
	void InitialSetting(string _imgPath, string _output_IxImgPath, string _output_IyImgPath, double _threshold);

	void Reshow(string name, Mat _img, double size);
	void HSVtoRGB(double h, double s, double v, double &r, double &g, double &b);

	void CalculateIxIy();
	void GaussianFiltering();
	void FindeResponseR();
	void FindHarrisCorner();
	void ShowCorners();

public:

	string imgPath;
	string output_IxImgPath, output_IyImgPath;

	Mat inputImg;
	Mat IxImg, IyImg;

	Mat Ix2Img, Iy2Img, IxyImg;
	Mat gIx2Img, gIy2Img, gIxyImg;
	Mat resRImg;

	Mat thrImg;

	vector <pair<int, int>> cornerLoc;

	int width, height;
	double threshold = 300;

	bool showFlag = true;
	bool axisAlignedCorner = false;

};