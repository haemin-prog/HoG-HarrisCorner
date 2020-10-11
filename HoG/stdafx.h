#pragma once


#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <algorithm>
#include <vector>

// 외부 라이브러리 로드
#if defined _M_IX86 // 32bit 빌드
#define _PLATFORM_VER 32
#ifdef _DEBUG // Debug 모드
#else // Release 모드
#endif

#elif defined _M_X64 // 64bit 빌드
#define _PLATFORM_VER 64
#ifdef _DEBUG // Debug 모드
#pragma comment(lib,"./Library/opencv/build/x64/vc14/lib/opencv_world420d.lib")
#else // Release 모드
#pragma comment(lib,"./Library/opencv/build/x64/vc14/lib/opencv_world420.lib")
#endif
#endif

#include "opencv.hpp"
#include "highgui.hpp"
#include "features2d.hpp"
#include "core.hpp"
#include "imgproc.hpp"

