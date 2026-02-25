// cuSIFT.cu

#include "cuSIFT.h"
#include <chrono>
#include <cmath> // For sqrt
#include <spdlog/spdlog.h>
#include "cudaImage.h"
#include "cudaSift.h"
#include "cudautils.h"
#include "geomFuncs_blas.hpp"

cudasift_settings CUDASIFT_DEFAULT_SETTINGS()
{
    cudasift_settings settings;
    // settings.max_num_features = 1000; // Removed
    settings.initial_gauss_blur = 1.0;
    settings.extract_sift_thresh = 2.0;
    settings.lowest_scale = 0.0;
    settings.num_octaves = 5;
    settings.find_homography_thresh = 5.0;
    settings.find_homography_min_score = 0.85;
    settings.find_homography_max_ambiguity = 0.95;
    settings.find_homography_max_iterations = 20000;
    settings.improve_homography_thresh = 3.5;
    settings.improve_homography_min_score = 0.85;
    settings.improve_homography_max_ambiguity = 0.95;
    settings.improve_homography_max_iterations = 1000;
    return settings;
};

static inline void pHomo(float *H)
{
    float val;
    spdlog::debug("Homography:");
    for (int i = 0; i < 3; i++)
    {
        std::string line = "\t|";
        for (int j = 0; j < 3; j++)
        {
            val = H[i * 3 + j];
            if (val == 0.0f)
                val = 0.0f;
            if (val < 0.0f)
                line = line + " " + std::to_string(val);
            else
                line = line + "  " + std::to_string(val);
        }
        line = line + " |";
        spdlog::debug(line);
    }
}

int CUDASIFT(
    const float *h_img1, int w1, int h1,
    const float *h_img2, int w2, int h2,
    float *homography,
    float *inlier_ratio,
    const struct cudasift_settings *settings)
{
    // int num_features = 1000; // Removed
    double initBlur = 1.0;
    float sift_thresh = 2.0f;
    float lowestScale = 0.0f;
    int find_homography_num_loops = 20000;
    float find_homography_min_score = 0.85f;
    float find_homography_max_ambiguity = 0.95f;
    float find_homography_thresh = 5.0f;
    int improve_homography_num_loops = 1000;
    float improve_homography_min_score = 0.85f;
    float improve_homography_max_ambiguity = 0.95f;
    float improve_homography_thresh = 3.5f;
    int num_octaves = 5;
    int num_matches = 0;
    int num_inliers = 0;
    float inlier = 0.0f;
    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;
    double elapsed_time = 0.0;
    const unsigned int max_width = (w1 > w2) ? w1 : w2;
    const unsigned int max_height = (h1 > h2) ? h1 : h2;
    SiftData siftData1, siftData2;
    CudaImage img1, img2;
    SiftTempMem tempMem;
    std::chrono::high_resolution_clock::time_point tstart, tend;
    tstart = std::chrono::high_resolution_clock::now();
    spdlog::debug("========================================");
    spdlog::debug("Initializing cudaSift");

    try
    {
        InitCuda();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        spdlog::debug("========================================");
        return CUSIFT_ERROR_NO_GPU_DEVICES;
    }

    if (settings)
    {
        // extract settings to local stack vars, and check for validity
        // num_features = settings->max_num_features; // Removed
        // num_features = (num_features < 1000) ? (1000) : (num_features); // Removed
        initBlur = settings->initial_gauss_blur;
        initBlur = (initBlur <= 1.0) ? (1.0) : (initBlur);
        sift_thresh = settings->extract_sift_thresh;
        sift_thresh = (sift_thresh <= 0.1) ? (0.1) : (sift_thresh);
        lowestScale = settings->lowest_scale;
        lowestScale = (lowestScale <= 0.0) ? (0.0) : (lowestScale);
        find_homography_num_loops = settings->find_homography_max_iterations;
        find_homography_num_loops = (find_homography_num_loops > 50000) ? (50000) : (find_homography_num_loops);
        find_homography_max_ambiguity = settings->find_homography_max_ambiguity;
        find_homography_min_score = settings->find_homography_min_score;
        find_homography_thresh = settings->find_homography_thresh;
        improve_homography_num_loops = settings->improve_homography_max_iterations;
        improve_homography_num_loops = (improve_homography_num_loops > 1000) ? (1000) : (improve_homography_num_loops);
        improve_homography_max_ambiguity = settings->improve_homography_max_ambiguity;
        improve_homography_min_score = settings->improve_homography_min_score;
        improve_homography_thresh = settings->improve_homography_thresh;
        num_octaves = settings->num_octaves;
        num_octaves = (num_octaves < 3) ? (3) : (num_octaves);
        num_octaves = (num_octaves > 6) ? (6) : (num_octaves);
    }

    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        img1.Allocate(w1, h1, iAlignUp(w1, 128), false, NULL, h_img1);
        img2.Allocate(w2, h2, iAlignUp(w2, 128), false, NULL, h_img2);
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch (const std::exception &e)
    {
        // ... (error handling)
        return CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY;
    }
    time_span = (t2 - t1);

    try
    {
        elapsed_time = img1.Download();
        elapsed_time += img2.Download();
    }
    catch (const std::exception &e)
    {
        // ... (error handling)
        return CUSIFT_ERROR_COPYING_TO_DEVICE;
    }
    spdlog::debug("CUDA images downloaded to device {}", (elapsed_time / 1000.0) + time_span.count());

    // Dynamically calculate max number of features and initialize SiftData
    int num_features1 = 5 * sqrt((double)w1 * h1);
    num_features1 = (num_features1 < 1000) ? (1000) : (num_features1);
    int num_features2 = 5 * sqrt((double)w2 * h2);
    num_features2 = (num_features2 < 1000) ? (1000) : (num_features2);

    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        InitSiftData(siftData1, num_features1, true, true);
        InitSiftData(siftData2, num_features2, true, true);
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch (const std::exception &e)
    {
        // ... (error handling)
        return CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY;
    }
    time_span = (t2 - t1);
    spdlog::debug("SiftData initialized {}", time_span.count());

    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        tempMem.Allocate(max_width, max_height, num_octaves);
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch (const std::exception &e)
    {
        // ... (error handling)
        return CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY;
    }
    time_span = (t2 - t1);

    try
    {
        elapsed_time = ExtractSift(siftData1, img1, num_octaves, initBlur, sift_thresh, lowestScale, tempMem.get_device_pointer());
    }
    catch (const std::exception &e)
    {
        // ... (error handling)
        return CUSIFT_ERROR_EXTRACTING_FEATURES;
    }
    spdlog::debug("Sift features extracted from image 1 {}", elapsed_time / 1000.0);

    try
    {
        elapsed_time = ExtractSift(siftData2, img2, num_octaves, initBlur, sift_thresh, lowestScale, tempMem.get_device_pointer());
    }
    catch (const std::exception &e)
    {
        // ... (error handling)
        return CUSIFT_ERROR_EXTRACTING_FEATURES;
    }
    tempMem.clear();
    spdlog::debug("Sift features extracted from image 2 {}", elapsed_time / 1000.0);
    spdlog::debug("Image1 SiftFeatures: {}/{}", siftData1.numPts, num_features1);
    spdlog::debug("Image2 SiftFeatures: {}/{}", siftData2.numPts, num_features2);

    if (siftData1.numPts < 20 || siftData2.numPts < 20)
    {
        spdlog::debug("Not enough SiftFeatures to attempt matching");
        spdlog::debug("CudaSift Failed");
        spdlog::debug("========================================\\n");
        return CUSIFT_ERROR_EXTRACTING_FEATURES;
    }

    try
    {
        elapsed_time = MatchSiftData(siftData1, siftData2);
    }
    catch (const std::exception &e)
    {
        // ... (error handling)
        return CUSIFT_ERROR_MATCHING_FEATURES;
    }
    spdlog::debug("Sift features matched {}", elapsed_time / 1000.0);
    siftData2.Free();

    try
    {
        elapsed_time = FindHomography(siftData1, homography, &num_matches, find_homography_num_loops, find_homography_min_score, find_homography_max_ambiguity, find_homography_thresh);
    }
    catch (const std::exception &e)
    {
        // ... (error handling)
        return CUSIFT_ERROR_COMPUTING_HOMOGRAPHY;
    }
    spdlog::debug("Homography found {}", elapsed_time / 1000.0);
    spdlog::debug("Number of matches: {}", num_matches);

    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        num_inliers = ImproveHomography_Mat(siftData1, homography, improve_homography_num_loops, improve_homography_min_score, improve_homography_max_ambiguity, improve_homography_thresh);
        t2 = std::chrono::high_resolution_clock::now();
        pHomo(homography);
    }
    catch (const std::exception &e)
    {
        // ... (error handling)
        return CUSIFT_ERROR_IMPROVING_HOMOGRAPHY;
    }
    time_span = (t2 - t1);
    spdlog::debug("Homography improved {}", time_span.count());
    spdlog::debug("Number of inliers: {}", num_inliers);
    spdlog::debug("Inlier ratio: {}%%", 100.0 * (num_inliers / (double)siftData1.numPts));
    inlier = (float)num_inliers / (float)siftData1.numPts;
    if (inlier_ratio != nullptr)
    {
        *inlier_ratio = inlier;
    }
    tend = std::chrono::high_resolution_clock::now();
    time_span = (tend - tstart);
    spdlog::debug("CudaSift Completed {}", time_span.count());
    spdlog::debug("========================================\\n");
    return inlier > 0.0 ? CUSIFT_ERROR_NONE : CUSIFT_ERROR_NO_INLIERS;
}
