#pragma once
#ifndef __GEOMFUNCS_MAT_HPP__
#define __GEOMFUNCS_MAT_HPP__

#include "cudaSift.h"

int ImproveHomography_Mat(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

#endif // __GEOMFUNCS_MAT_HPP__
