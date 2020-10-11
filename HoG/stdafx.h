#pragma once


#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <algorithm>
#include <vector>

// �ܺ� ���̺귯�� �ε�
#if defined _M_IX86 // 32bit ����
#define _PLATFORM_VER 32
#ifdef _DEBUG // Debug ���
#else // Release ���
#endif

#elif defined _M_X64 // 64bit ����
#define _PLATFORM_VER 64
#ifdef _DEBUG // Debug ���
#pragma comment(lib,"./Library/opencv/build/x64/vc14/lib/opencv_world420d.lib")
#else // Release ���
#pragma comment(lib,"./Library/opencv/build/x64/vc14/lib/opencv_world420.lib")
#endif
#endif

#include "opencv.hpp"
#include "highgui.hpp"
#include "features2d.hpp"
#include "core.hpp"
#include "imgproc.hpp"

