//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                        //
//********************************************************//

/* C library functions */
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <ctime>

/* C++ library functions */
#include <stdexcept>

#include "cudasift_export.h"

/* Cuda runtime API */
// #include <cuda.h>
// #include <cuda_runtime.h>

/* cuda sift includes */
#include "geomFuncs.h"

/* Matlab mex header file */
#include "cudaSift.h"


#define TRY(stuff) do {                                 \
  Log("try ----- " #stuff);                             \
  clock_t m_time = clock();                             \
  double els_time = 0.0;                                \
  try                                                   \
  {                                                     \
    stuff;                                              \
  }                                                     \
  catch (const std::exception& e)                       \
  {                                                     \
    fprintf(stderr, "[LOG]: caught -- %s\n", e.what()); \
    Flag_Exit = true;                                   \
  }                                                     \
  els_time = (clock()-m_time)/(double)CLOCKS_PER_SEC;   \
  fprintf(stdout, "                  %lfs\n", els_time);\
  if (Flag_Exit) goto exit_failure;                     \
} while (0)



/* Computational routine */
CUDASIFT_EXPORT 
void CUDASIFT(int32_t      devNum,
              float*       image1,     
              int32_t      image1_rows,
              int32_t      image1_cols,
              float*       image2,     
              int32_t      image2_rows,
              int32_t      image2_cols,
              int32_t      siftPoints,
              float        initBlur,
              float        thresh,
              double **    ret_homography,
              void (*SharePointer_Mx)(void* siftData))
{
  bool Flag_Exit = false;
  float * TempMem = NULL;
  CudaImage img1, img2;

  for (int i = 0; i < 9; i += 3) {
    (*ret_homography)[i + 0] = 0.0;
    (*ret_homography)[i + 1] = 0.0;
    (*ret_homography)[i + 2] = 0.0;
  }

  #define Log(x) fprintf(stdout, "[LOG]: %s\n", x)

  /* Initialize Cuda */
  TRY(InitCuda((int)devNum));


  TRY(img1.Allocate(image1_cols, image1_rows, iAlignUp(image1_cols, 128), false, NULL, image1));
  TRY(img1.Download());

  TRY(img2.Allocate(image2_cols, image2_rows, iAlignUp(image2_cols, 128), false, NULL, image2));
  TRY(img2.Download());

  /* Extract Sift features from images */
  SiftData siftData1;
  TempMem = NULL;
  TRY(InitSiftData(siftData1, siftPoints, true, true));
  TRY(TempMem = AllocSiftTempMemory(img1.width, img1.height, 5, false));
  TRY(ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f, false, TempMem));
  FreeSiftTempMemory(TempMem);
  img1.~CudaImage();

  if (SharePointer_Mx)
    SharePointer_Mx((void*)&siftData1);

  SiftData siftData2;
  TempMem = NULL;
  TRY(InitSiftData(siftData2, siftPoints, true, true));
  TRY(TempMem = AllocSiftTempMemory(img2.width, img2.height, 5, false));
  TRY(ExtractSift(siftData2, img2, 5, initBlur, thresh, 0.0f, false, TempMem));
  FreeSiftTempMemory(TempMem);
  img2.~CudaImage();

  /* Match Sift features */
  TRY(MatchSiftData(siftData1, siftData2));

  FreeSiftData(siftData2);

  /* Find Homography */
  float homography[9];  int numMatches;
  TRY(FindHomography(siftData1, homography, &numMatches, 100000, 0.85f, 0.95f, thresh));
  TRY(ImproveHomography(siftData1, homography, 500, 0.00f, 0.95f, thresh));


  FreeSiftData(siftData1);

  for (int i = 0; i < 9; i += 3) {
    (*ret_homography)[i + 0] = homography[i + 0];
    (*ret_homography)[i + 1] = homography[i + 1];
    (*ret_homography)[i + 2] = homography[i + 2];
  }

exit_success:
  Log("Exit ---- Success\n\n");
  goto exit;

exit_failure:
  Log("Exit ---- Failure\n\n");

exit:
  //cudaDeviceReset();
  return;
}
