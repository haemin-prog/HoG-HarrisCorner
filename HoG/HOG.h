#pragma once

#include "stdafx.h"

using namespace std;
using namespace cv;

class HOG {

public:
	void InitialSetting(string _imgPath, string _output_MagnitudePath, string _output_GradientPath, string _output_resultPath);

	void SelectBoundingBox();

	void CalculateGradient();
	void SaveHistoVector(double val);
	void DrawHistoVector();

	void HSVtoRGB(double _r, double _g, double _b, double &_h, double &_s, double &_v);

	void Reshow(string name, Mat _img, double size);

public:
	string imgPath;
	string output_MagnitudePath, output_directionPath, output_resultPath;

	Mat inputImg;
	Mat boundBoxImg;
	Mat magnitudeImg, directionImg;
	Mat drawImg;

	int histoVector[12] = { 0 };
	double normalVector[12] = { 0 };

	pair<int, int> leftTopCorner, rightBottomCorner;

	int oriWidth, oriHeight;
	int boundWidth, boundHeight;

	bool showFlag = true;

}; 
