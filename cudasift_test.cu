

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

#include "cuSIFT.h"

static void read_newline_separated_file(const char *file, uint32_t &width, uint32_t &height, std::vector<float> &img);
static inline void pH(float *H);

int main(int argc, char **argv)
{
    std::vector<float> im1;
    std::vector<float> im2;
    uint32_t w1 = 0;
    uint32_t h1 = 0;
    uint32_t w2 = 0;
    uint32_t h2 = 0;
    std::vector<float> H;
    float inlier_ratio = 0.0f;
    struct cudasift_settings settings;
    int res = 0;

    settings.max_num_features = 20000;
    settings.initial_gauss_blur = 1.0;
    settings.extract_sift_thresh = 2.0f;
    settings.lowest_scale = 0.0f;
    settings.num_octaves = 5;
    settings.find_homography_thresh = 5.0f;
    settings.find_homography_min_score = 0.85f;
    settings.find_homography_max_ambiguity = 0.95f;
    settings.find_homography_max_iterations = 50000;
    settings.improve_homography_thresh = 3.5f;
    settings.improve_homography_min_score = 0.85f;
    settings.improve_homography_max_ambiguity = 0.95f;
    settings.improve_homography_max_iterations = 1000;

    // Expecting two arguments ./foo [img1] [img2]
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " [img1_file] [img2_file]\n";
        exit(EXIT_FAILURE);
    }

    read_newline_separated_file(argv[1], w1, h1, im1);
    read_newline_separated_file(argv[2], w2, h2, im2);

    H.resize(9);

    res = CUDASIFT(
        im1.data(), w1, w2,
        im2.data(), w2, h2,
        H.data(), &inlier_ratio,
        &settings);

    // Print H
    pH(H.data());
    fprintf(stdout, "Inlier Ratio: %f\n", inlier_ratio);
    fprintf(stdout, "Result: %d\n", res);

    return 0;
}

static void read_newline_separated_file(const char *file, uint32_t &width, uint32_t &height, std::vector<float> &img)
{
    std::ifstream fid(file);
    std::string line;
    size_t num_elements = 0;
    size_t idx = 0;

    img.clear();

    if (fid.is_open())
    {
        // First two lines are the width and height
        if (std::getline(fid, line))
        {
            width = static_cast<uint32_t>(std::stoul(line));
        }
        else
        {
            std::cerr << "Failed to parse the width of the input image: " << file << "\n";
            exit(EXIT_FAILURE);
        }

        if (std::getline(fid, line))
        {
            height = static_cast<uint32_t>(std::stoul(line));
        }
        else
        {
            std::cerr << "Failed to parse the height of the input image: " << file << "\n";
        }

        num_elements = (size_t)width * (size_t)height;
        img.reserve(num_elements);

        for (idx = 0; idx < num_elements; idx++)
        {
            if (std::getline(fid, line))
            {
                img.push_back(std::stof(line));
            }
            else
            {
                std::cerr << "Failed to parse expected elements in file: " << width << "x" << height << "\n";
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        std::cerr << "Failed to open file for reading: " << file << "\n";
        exit(EXIT_FAILURE);
    }
}

static inline void pH(float *H)
{
    float val;
    fprintf(stdout, "Homography:\n");
    for (int i = 0; i < 3; i++)
    {
        fprintf(stdout, "\t|");
        for (int j = 0; j < 3; j++)
        {
            val = H[i * 3 + j];
            if (val == 0.0f)
                val = 0.0f;
            if (val < 0.0f)
                fprintf(stdout, " %.9e", val);
            else
                fprintf(stdout, "  %.9e", val);
        }
        fprintf(stdout, " |\n");
    }
}