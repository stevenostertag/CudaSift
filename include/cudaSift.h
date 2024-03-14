#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"
#include "cudasift_export.h"

typedef struct {
  float xpos;
  float ypos;   
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float empty[3];
  float data[128];
} SiftPoint;

typedef struct {
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data;  // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
#endif
} SiftData;

void InitCuda(int devNum = 0);
float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp);
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, bool scaleUp = false, float *tempMemory = 0);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);
double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#include <stdint.h>

/**
 * @brief Generate a Homography of the two images to create a mosaic. The mosaic creation
 * should be outsourced to vl_fleat.
 * How to pass image data in from matlab.
 * img = imread("image.png");
 * img = rgb2gray(img);
 * img = img';
 * img = single(img); % img values are now ready for CUDASIFT
 * 
 * 
 * @param devNum (0) default, But if multiple GPUs, can choose between them with this var.
 * @param image1 float*, {single array matlab} !!ROW MAJOR!! single channel gray scale pixel values.
 * @param image1_rows rows in image1
 * @param image1_cols cols in image1
 * @param image2 float*, {single array matlab} !!ROW MAJOR!! single channel gray scale pixel values.
 * @param image2_rows rows in image2
 * @param image2_cols cols in image2
 * @param siftPoints Allocatable sift points. 10 * sqrt(numel(image1) + numel(image2)) is probably enough.
 * @param initBlur 4.0 is default, this value cna be moved around to try and get better results.
 * @param thresh 1.0 is default, this value cna be moved around to try and get better results.
 * @param homography Pointer to an allocated double array that can store 9 double values.
 */
 
CUDASIFT_EXPORT
void CUDASIFT(int32_t       devNum,
              float*        image1,
              int32_t       image1_rows,
              int32_t       image1_cols,
              float*        image2,
              int32_t       image2_rows,
              int32_t       image2_cols,
              int32_t       siftPoints,
              float         initBlur,
              float         thresh,
              double **     homography,
              void (*func)(void*));

#endif // CUDASIFT_H
