#ifndef __GEOMFUNCS_H
#define __GEOMFUNCS_H


#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);



#endif
