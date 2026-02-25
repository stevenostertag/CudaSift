#include "cuSIFT.h"
#include "cudaImage.h"
#include "cudaSift.h"
#include "geomFuncs_blas.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <spdlog/spdlog.h>

// --- Helper Function Prototypes ---
bool loadAndConvertToFloat(const std::string &path, cv::Mat &out_img);
void convertToOpencvKeypoints(const SiftData &siftData, std::vector<cv::KeyPoint> &keypoints);
void convertToOpencvMatches(const SiftData &siftData1, std::vector<cv::DMatch> &matches, float minScore);
int DebugCUDASIFT(
    const float *h_img1, int w1, int h1,
    const float *h_img2, int w2, int h2,
    const struct cudasift_settings *settings);

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

// --- Main Driver ---
int main(int argc, char *argv[])
{
    spdlog::set_level(spdlog::level::debug);
    spdlog::info("Starting CUDASIFT Visual Debugger...");

    if (argc != 3)
    {
        spdlog::error("Usage: {} <path_to_image_1> <path_to_image_2>", argv[0]);
        return -1;
    }

    std::string img1_path = argv[1];
    std::string img2_path = argv[2];

    spdlog::info("Loading and converting images...");

    cv::Mat img1_float, img2_float;

    if (!loadAndConvertToFloat(img1_path, img1_float) || !loadAndConvertToFloat(img2_path, img2_float))
    {
        spdlog::error("Halting due to image loading failure.");
        return -1;
    }

    cudasift_settings settings = CUDASIFT_DEFAULT_SETTINGS();
    settings.extract_sift_thresh = 1.3;

    spdlog::info("Image 1 Size: {}x{}", img1_float.cols, img1_float.rows);
    spdlog::info("Image 2 Size: {}x{}", img2_float.cols, img2_float.rows);

    DebugCUDASIFT(
        (float *)img1_float.data, img1_float.cols, img1_float.rows,
        (float *)img2_float.data, img2_float.cols, img2_float.rows,
        &settings);

    return 0;
}

// --- Function Implementations ---

bool loadAndConvertToFloat(const std::string &path, cv::Mat &out_img)
{
    cv::Mat loaded_image = cv::imread(path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if (loaded_image.empty())
    {
        spdlog::error("Failed to load image: {}", path);
        return false;
    }
    cv::Mat gray_image;
    if (loaded_image.channels() == 3)
    {
        cv::cvtColor(loaded_image, gray_image, cv::COLOR_BGR2GRAY);
    }
    else if (loaded_image.channels() == 4)
    {
        cv::cvtColor(loaded_image, gray_image, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        gray_image = loaded_image;
    }
    int depth = gray_image.depth();
    if (depth == CV_8U)
    {
        gray_image.convertTo(out_img, CV_32F, 1.0 / 255.0);
    }
    else if (depth == CV_16U)
    {
        gray_image.convertTo(out_img, CV_32F, 1.0 / 65535.0);
    }
    else if (depth == CV_32F)
    {
        out_img = gray_image;
    }
    else
    {
        spdlog::error("Unsupported image depth: {} for image {}", depth, path);
        return false;
    }
    return true;
}

void convertToOpencvKeypoints(const SiftData &siftData, std::vector<cv::KeyPoint> &keypoints)
{
    keypoints.clear();
    for (int i = 0; i < siftData.numPts; i++)
    {
        cv::KeyPoint kp;
        kp.pt.x = siftData.h_data[i].xpos;
        kp.pt.y = siftData.h_data[i].ypos;
        kp.size = siftData.h_data[i].scale;
        kp.angle = siftData.h_data[i].orientation;
        kp.response = siftData.h_data[i].score;
        keypoints.push_back(kp);
    }
}

void convertToOpencvMatches(const SiftData &siftData1, std::vector<cv::DMatch> &matches, float minScore)
{
    matches.clear();
    for (int i = 0; i < siftData1.numPts; ++i)
    {
        if (siftData1.h_data[i].match != -1 && siftData1.h_data[i].score >= minScore)
        {
            cv::DMatch m;
            m.queryIdx = i;
            m.trainIdx = siftData1.h_data[i].match;
            m.distance = 1.0f - siftData1.h_data[i].score;
            matches.push_back(m);
        }
    }
}

int DebugCUDASIFT(
    const float *h_img1, int w1, int h1,
    const float *h_img2, int w2, int h2,
    const struct cudasift_settings *settings)
{
    auto wait_for_input = []()
    {
        while (true)
        {
            int key = cv::waitKey(0);
            if (key == 'n')
                return true;
            if (key == 'q')
                return false;
        }
    };

    // =================================================================================
    // 0. INITIALIZATION & DATA SETUP
    // =================================================================================
    spdlog::info("Initializing CUDA and preparing data structures...");

    // Create cv::Mat headers for the input float data (no data is copied)
    cv::Mat img1_float_mat(h1, w1, CV_32F, (void *)h_img1);
    cv::Mat img2_float_mat(h2, w2, CV_32F, (void *)h_img2);

    // ** FIX: NORMALIZE the float images to 8-bit for correct display **
    // This creates a separate, correctly scaled image for visualization purposes only.
    cv::Mat img1_display, img2_display;
    cv::normalize(img1_float_mat, img1_display, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(img2_float_mat, img2_display, 0, 255, cv::NORM_MINMAX, CV_8U);

    SiftData siftData1, siftData2;
    CudaImage img1_cuda, img2_cuda;
    SiftTempMem tempMem;

    int num_features1 = 5 * sqrt((double)w1 * h1);
    num_features1 = (num_features1 < 1000) ? (1000) : (num_features1);
    int num_features2 = 5 * sqrt((double)w2 * h2);
    num_features2 = (num_features2 < 1000) ? (1000) : (num_features2);

    try
    {
        InitCuda();
        const unsigned int max_width = (w1 > w2) ? w1 : w2;
        const unsigned int max_height = (h1 > h2) ? h1 : h2;
        img1_cuda.Allocate(w1, h1, iAlignUp(w1, 128), false, NULL, h_img1);
        img2_cuda.Allocate(w2, h2, iAlignUp(w2, 128), false, NULL, h_img2);
        img1_cuda.Download();
        img2_cuda.Download();
        InitSiftData(siftData1, num_features1, true, true);
        InitSiftData(siftData2, num_features2, true, true);
        tempMem.Allocate(max_width, max_height, settings->num_octaves);
    }
    catch (const std::exception &e)
    {
        spdlog::error("Initialization failed: {}", e.what());
        return -1;
    }

    // =================================================================================
    // 1. FEATURE EXTRACTION
    // =================================================================================
    spdlog::info("Step 1: Extracting SIFT features...");
    try
    {
        ExtractSift(siftData1, img1_cuda, settings->num_octaves, settings->initial_gauss_blur, settings->extract_sift_thresh, settings->lowest_scale, tempMem.get_device_pointer());
        ExtractSift(siftData2, img2_cuda, settings->num_octaves, settings->initial_gauss_blur, settings->extract_sift_thresh, settings->lowest_scale, tempMem.get_device_pointer());
    }
    catch (const std::exception &e)
    {
        spdlog::error("Feature extraction failed: {}", e.what());
        return CUSIFT_ERROR_EXTRACTING_FEATURES;
    }

    spdlog::info("Image 1 Features: {} | Image 2 Features: {}", siftData1.numPts, siftData2.numPts);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    convertToOpencvKeypoints(siftData1, keypoints1);
    convertToOpencvKeypoints(siftData2, keypoints2);
    cv::Mat img1_keypoints, img2_keypoints;
    cv::drawKeypoints(img1_display, keypoints1, img1_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(img2_display, keypoints2, img2_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow("Debug - Image 1 Keypoints", cv::WINDOW_NORMAL);
    cv::namedWindow("Debug - Image 2 Keypoints", cv::WINDOW_NORMAL);
    cv::imshow("Debug - Image 1 Keypoints", img1_keypoints);
    cv::imshow("Debug - Image 2 Keypoints", img2_keypoints);
    spdlog::info("Displaying keypoints. Press 'n' for next step, or 'q' to quit.");
    if (!wait_for_input())
    {
        cv::destroyAllWindows();
        return -1;
    }

    // =================================================================================
    // 2. FEATURE MATCHING
    // =================================================================================
    spdlog::info("Step 2: Matching features between images...");
    try
    {
        MatchSiftData(siftData1, siftData2);
    }
    catch (const std::exception &e)
    {
        spdlog::error("Feature matching failed: {}", e.what());
        return CUSIFT_ERROR_MATCHING_FEATURES;
    }

    std::vector<cv::DMatch> initial_matches;
    convertToOpencvMatches(siftData1, initial_matches, settings->find_homography_min_score);
    spdlog::info("Found {} initial matches (score > {:.2f})", initial_matches.size(), settings->find_homography_min_score);
    cv::Mat img_matches;
    cv::drawMatches(img1_display, keypoints1, img2_display, keypoints2, initial_matches, img_matches);
    cv::namedWindow("Debug - Initial Feature Matches", cv::WINDOW_NORMAL);
    cv::imshow("Debug - Initial Feature Matches", img_matches);
    spdlog::info("Displaying initial matches. Press 'n' for next step, or 'q' to quit.");
    if (!wait_for_input())
    {
        cv::destroyAllWindows();
        return -1;
    }

    // =================================================================================
    // 3. HOMOGRAPHY ESTIMATION (RANSAC)
    // =================================================================================
    spdlog::info("Step 3: Finding homography with RANSAC...");
    float homography[9];
    int numMatches = 0;
    try
    {
        FindHomography(siftData1, homography, &numMatches, settings->find_homography_max_iterations, settings->find_homography_min_score, settings->find_homography_max_ambiguity, settings->find_homography_thresh);
    }
    catch (const std::exception &e)
    {
        spdlog::error("Homography computation failed: {}", e.what());
        return CUSIFT_ERROR_COMPUTING_HOMOGRAPHY;
    }

    cv::Mat H = cv::Mat(3, 3, CV_32F, homography).clone();
    std::vector<cv::DMatch> inlier_matches;
    if (numMatches > 0)
    {
        spdlog::info("RANSAC found {} inlier matches.", numMatches);
        std::vector<cv::Point2f> points1, points2;
        for (const auto &match : initial_matches)
        {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
        std::vector<uchar> inlier_mask;
        cv::findHomography(points1, points2, cv::RANSAC, settings->find_homography_thresh, inlier_mask);
        for (size_t i = 0; i < inlier_mask.size(); ++i)
        {
            if (inlier_mask[i])
            {
                inlier_matches.push_back(initial_matches[i]);
            }
        }
    }
    else
    {
        spdlog::warn("RANSAC found 0 inliers. The homography is likely invalid.");
    }
    cv::Mat img_inliers;
    cv::drawMatches(img1_display, keypoints1, img2_display, keypoints2, inlier_matches, img_inliers);
    cv::namedWindow("Debug - RANSAC Inlier Matches", cv::WINDOW_NORMAL);
    cv::imshow("Debug - RANSAC Inlier Matches", img_inliers);
    spdlog::info("Displaying RANSAC inliers. Press 'n' for next step, or 'q' to quit.");
    if (!wait_for_input())
    {
        cv::destroyAllWindows();
        return -1;
    }

    // =================================================================================
    // 4. FINAL IMPROVEMENT & CLEANUP
    // =================================================================================
    int numInliersFinal = 0;
    try
    {
        numInliersFinal = ImproveHomography_Mat(siftData1, homography, settings->improve_homography_max_iterations, settings->improve_homography_min_score, settings->improve_homography_max_ambiguity, settings->improve_homography_thresh);
    }
    catch (const std::exception &e)
    {
        spdlog::error("Improve homography failed: {}", e.what());
        return CUSIFT_ERROR_IMPROVING_HOMOGRAPHY;
    }

    float inlier_ratio = (numInliersFinal > 0) ? (float)numInliersFinal / (float)siftData1.numPts : 0.0f;
    spdlog::info("Final refined homography has {} inliers (Inlier Ratio: {:.2f}%)", numInliersFinal, inlier_ratio * 100.0f);

    siftData1.Free();
    siftData2.Free();
    tempMem.clear();
    cv::destroyAllWindows();

    pHomo(homography);

    spdlog::info("Debugger finished.");

    return (numInliersFinal > 0) ? CUSIFT_ERROR_NONE : CUSIFT_ERROR_NO_INLIERS;
}
