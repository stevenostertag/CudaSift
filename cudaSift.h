#pragma once
#ifndef CUDASIFT_H
#define CUDASIFT_H

#include <stddef.h>
#include <texture_types.h>

#include "cudaImage.h"

#define NUM_SCALES 5

// Scale down thread block width
#define SCALEDOWN_W 64 // 60

// Scale down thread block height
#define SCALEDOWN_H 16 // 8

// Scale up thread block width
#define SCALEUP_W 64

// Scale up thread block height
#define SCALEUP_H 8

// Find point thread block width
#define MINMAX_W 30 // 32

// Find point thread block height
#define MINMAX_H 8 // 16

// Laplace thread block width
#define LAPLACE_W 128 // 56

// Laplace rows per thread
#define LAPLACE_H 4

// Number of laplace scales
#define LAPLACE_S (NUM_SCALES + 3)

// Laplace filter kernel radius
#define LAPLACE_R 4

#define LOWPASS_W 24 // 56
#define LOWPASS_H 32 // 16
#define LOWPASS_R 4

typedef struct
{
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

class SiftData
{
public:
    int numPts; // Number of available Sift points
    int maxPts; // Number of allocated Sift points

    SiftPoint *h_data; // Host (CPU) data
    SiftPoint *d_data; // Device (GPU) data

    SiftData() : numPts(0), maxPts(0), h_data(0), d_data(0) {};
    ~SiftData() { Free(); }

    void Free();
    void Allocate(int num = 1024, bool host = false, bool dev = true);
};

class SiftTempMem
{
private:
    float *m_temp_device_ptr;

public:
    SiftTempMem();
    SiftTempMem(int width, int height, int numOctaves);
    ~SiftTempMem();
    void Allocate(int width, int height, int numOctaves);
    float *get_device_pointer();
    void clear();
};

void InitCuda(int devNum = 0);
double ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, float *tempMemory = 0);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);
double FindHomography(SiftData &data, float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);
void saveSiftData(const SiftData &dat1, const SiftData &data2, const char *file);

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp);
double ScaleDown(CudaImage &res, CudaImage &src, float variance);
double ScaleUp(CudaImage &res, CudaImage &src);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
double RescalePositions(SiftData &siftData, float scale);
double LowPass(CudaImage &res, CudaImage &src, float scale);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave);
double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave);

#endif
