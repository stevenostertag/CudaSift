#ifndef __CUSIFT_HPP__
#define __CUSIFT_HPP__

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define CUSIFT_ERROR_NONE 0
#define CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY -1
#define CUSIFT_ERROR_COPYING_TO_DEVICE -2
#define CUSIFT_ERROR_COPYING_FROM_DEVICE -3
#define CUSIFT_ERROR_EXTRACTING_FEATURES -4
#define CUSIFT_ERROR_MATCHING_FEATURES -5
#define CUSIFT_ERROR_COMPUTING_HOMOGRAPHY -6
#define CUSIFT_ERROR_IMPROVING_HOMOGRAPHY -7
#define CUSIFT_ERROR_NO_INLIERS -8
#define CUSIFT_ERROR_NO_GPU_DEVICES -9

    struct cudasift_settings
    {
        // Default
        float initial_gauss_blur;               // 1.0
        float extract_sift_thresh;              // 2.0f
        float lowest_scale;                     // 0.0f
        int num_octaves;                        // 5
        float find_homography_thresh;           // 5.0f
        float find_homography_min_score;        // 0.85f
        float find_homography_max_ambiguity;    // 0.95f
        int find_homography_max_iterations;     // 20,000
        float improve_homography_thresh;        // 3.5f
        float improve_homography_min_score;     // 0.85f
        float improve_homography_max_ambiguity; // 0.95f
        int improve_homography_max_iterations;  // 1,000
    };

    EXPORT cudasift_settings CUDASIFT_DEFAULT_SETTINGS();

    /**
     * @brief Use CUDA to compute SIFT features for two images, and return the homography matrix.
     * The homography matrix is a 3x3 matrix that maps points in the second image to points in the first image.
     *
     * @param h_img1 Host pointer to the first image.
     * @param w1 Width of the first image. Fastest dimension.
     * @param h1 Height of the first image. Slowest dimension.
     * @param h_img2 Host pointer to the second image.
     * @param w2 Width of the second image. Fastest dimension.
     * @param h2 Height of the second image. Slowest dimension.
     * @param homography Float array that can hold at least 9 elements. Row major (3x3)
     * @param inlier_ratio If not NULL, stores the inlier ratio from RANSAC
     * @param settings cudasift settings struct.
     */
    EXPORT int CUDASIFT(
        const float *h_img1, int w1, int h1,
        const float *h_img2, int w2, int h2,
        float *homography,
        float *inlier_ratio,
        const struct cudasift_settings *settings);

#ifdef __cplusplus
}
#endif

#endif // __CUSIFT_HPP__
