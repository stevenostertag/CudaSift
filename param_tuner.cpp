#include "cuSIFT.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <map>
#include <chrono>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <cmath> // For std::sqrt

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

// --- Configuration & Data Structs ---

// ** NEW **: A generic struct to define a range for any parameter.
template <typename T>
struct Range
{
    T start;
    T end;
    int count = 0;
};

// ** MODIFIED **: Vectors replaced with Range structs.
struct TunerConfig
{
    std::string image_dir;
    std::string output_csv;
    float shearing_thresh = 0.2f;
    float scaling_thresh = 0.2f;

    Range<double> initial_gauss_blur_range;
    Range<float> extract_sift_thresh_range;
    Range<int> num_octaves_range;
    Range<float> find_homography_thresh_range;
};

struct ParamKey
{
    double initial_gauss_blur;
    float extract_sift_thresh;
    int num_octaves;
    float find_homography_thresh;

    bool operator<(const ParamKey &other) const
    {
        return std::tie(initial_gauss_blur, extract_sift_thresh, num_octaves, find_homography_thresh) <
               std::tie(other.initial_gauss_blur, other.extract_sift_thresh, other.num_octaves, other.find_homography_thresh);
    }
};

struct AggregatedResult
{
    int pass_count = 0;
    int total_runs = 0;
    double total_inlier_ratio = 0.0;
};

// --- Helper Functions ---

// ** MODIFIED **: Parses range arguments (start, end, count) instead of explicit lists.
void parse_args(int argc, char *argv[], TunerConfig &config)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--image_dir")
        {
            config.image_dir = argv[++i];
        }
        else if (arg == "--output_csv")
        {
            config.output_csv = argv[++i];
        }
        else if (arg == "--shearing_thresh")
        {
            config.shearing_thresh = std::stof(argv[++i]);
        }
        else if (arg == "--scaling_thresh")
        {
            config.scaling_thresh = std::stof(argv[++i]);
        }
        else if (arg == "--initial_gauss_blur_range")
        {
            if (i + 3 >= argc)
            {
                spdlog::error("Usage: --initial_gauss_blur_range <start> <end> <count>");
                exit(1);
            }
            config.initial_gauss_blur_range = {std::stod(argv[i + 1]), std::stod(argv[i + 2]), std::stoi(argv[i + 3])};
            i += 3;
        }
        else if (arg == "--extract_sift_thresh_range")
        {
            if (i + 3 >= argc)
            {
                spdlog::error("Usage: --extract_sift_thresh_range <start> <end> <count>");
                exit(1);
            }
            config.extract_sift_thresh_range = {std::stof(argv[i + 1]), std::stof(argv[i + 2]), std::stoi(argv[i + 3])};
            i += 3;
        }
        else if (arg == "--num_octaves_range")
        {
            if (i + 3 >= argc)
            {
                spdlog::error("Usage: --num_octaves_range <start> <end> <count>");
                exit(1);
            }
            config.num_octaves_range = {std::stoi(argv[i + 1]), std::stoi(argv[i + 2]), std::stoi(argv[i + 3])};
            i += 3;
        }
        else if (arg == "--find_homography_thresh_range")
        {
            if (i + 3 >= argc)
            {
                spdlog::error("Usage: --find_homography_thresh_range <start> <end> <count>");
                exit(1);
            }
            config.find_homography_thresh_range = {std::stof(argv[i + 1]), std::stof(argv[i + 2]), std::stoi(argv[i + 3])};
            i += 3;
        }
    }
}

std::vector<std::pair<std::string, std::string>> find_image_pairs(const std::string &dir_path)
{
    std::map<int, std::vector<std::string>> pair_map;
    for (const auto &entry : fs::directory_iterator(dir_path))
    {
        if (entry.path().extension() == ".tiff" || entry.path().extension() == ".tif")
        {
            std::string filename = entry.path().stem().string();
            size_t last_underscore = filename.find_last_of('_');
            if (last_underscore != std::string::npos)
            {
                try
                {
                    int pair_id = std::stoi(filename.substr(last_underscore + 1));
                    pair_map[pair_id].push_back(entry.path().string());
                }
                catch (const std::exception &e)
                { /* Ignore */
                }
            }
        }
    }
    std::vector<std::pair<std::string, std::string>> pairs;
    for (auto const &[id, files] : pair_map)
    {
        if (files.size() == 2)
        {
            pairs.push_back({files[0], files[1]});
        }
        else
        {
            spdlog::warn("Found {} files for pair ID {}, expected 2. Skipping.", files.size(), id);
        }
    }
    return pairs;
}

// ** MODIFIED **: Generates discrete values from ranges, then creates the parameter sets.
std::vector<cudasift_settings> generate_parameter_sets(const TunerConfig &config)
{
    // Helper lambda to generate values from a range
    auto generate_values = []<typename T>(const Range<T> &range)
    {
        std::vector<T> values;
        if (range.count > 0)
        {
            if (range.count == 1)
            {
                values.push_back(range.start);
            }
            else
            {
                for (int i = 0; i < range.count; ++i)
                {
                    double step = static_cast<double>(i) / (range.count - 1);
                    T value = range.start + static_cast<T>((range.end - range.start) * step);
                    values.push_back(value);
                }
            }
        }
        return values;
    };

    auto blur_values = generate_values(config.initial_gauss_blur_range);
    auto sift_thresh_values = generate_values(config.extract_sift_thresh_range);
    auto octaves_values = generate_values(config.num_octaves_range);
    auto homo_thresh_values = generate_values(config.find_homography_thresh_range);

    // If any range was not specified, add a default value to ensure the loop runs
    if (blur_values.empty())
        blur_values.push_back(CUDASIFT_DEFAULT_SETTINGS().initial_gauss_blur);
    if (sift_thresh_values.empty())
        sift_thresh_values.push_back(CUDASIFT_DEFAULT_SETTINGS().extract_sift_thresh);
    if (octaves_values.empty())
        octaves_values.push_back(CUDASIFT_DEFAULT_SETTINGS().num_octaves);
    if (homo_thresh_values.empty())
        homo_thresh_values.push_back(CUDASIFT_DEFAULT_SETTINGS().find_homography_thresh);

    std::vector<cudasift_settings> sets;
    for (double blur : blur_values)
    {
        for (float sift_thresh : sift_thresh_values)
        {
            for (int octaves : octaves_values)
            {
                for (float homo_thresh : homo_thresh_values)
                {
                    cudasift_settings s = CUDASIFT_DEFAULT_SETTINGS();
                    s.initial_gauss_blur = blur;
                    s.extract_sift_thresh = sift_thresh;
                    s.num_octaves = octaves;
                    s.find_homography_thresh = homo_thresh;
                    sets.push_back(s);
                }
            }
        }
    }
    return sets;
}

bool load_and_prepare_image(const std::string &path, float *&h_img, int &w, int &h)
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
    cv::Mat out_img;
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

    w = out_img.cols;
    h = out_img.rows;
    h_img = new float[w * h];
    memcpy(h_img, out_img.ptr<float>(0), w * h * sizeof(float));
    return true;
}

bool CheckHomography(float *H, float shearing_thresh, float scaling_thresh)
{
    if (std::abs(1.0f - H[0]) > scaling_thresh)
    {
        return false;
    }
    if (std::abs(1.0f - H[4]) > scaling_thresh)
    {
        return false;
    }
    if (std::abs(H[1]) > shearing_thresh)
    {
        return false;
    }
    if (std::abs(H[3]) > shearing_thresh)
    {
        return false;
    }
    return true;
}

void analyze_results(const std::string &csv_path)
{
    spdlog::info("--- Starting Analysis of Results ---");
    std::ifstream file(csv_path);
    if (!file.is_open())
    {
        spdlog::error("Could not open results file for analysis: {}", csv_path);
        return;
    }

    std::map<ParamKey, AggregatedResult> analysis_map;
    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(ss, field, ','))
            fields.push_back(field);

        bool passed = (fields[2] == "true");
        float inlier_ratio = std::stof(fields[3]);
        ParamKey key;
        key.initial_gauss_blur = std::stod(fields[9]);
        key.extract_sift_thresh = std::stof(fields[10]);
        key.num_octaves = std::stoi(fields[11]);
        key.find_homography_thresh = std::stof(fields[12]);
        analysis_map[key].total_runs++;
        if (passed)
        {
            analysis_map[key].pass_count++;
            analysis_map[key].total_inlier_ratio += inlier_ratio;
        }
    }
    file.close();

    ParamKey best_params;
    float best_pass_rate = -1.0f;
    float best_avg_inlier_ratio = -1.0f;

    for (const auto &[key, result] : analysis_map)
    {
        if (result.total_runs == 0)
            continue;
        float pass_rate = static_cast<float>(result.pass_count) / result.total_runs;
        float avg_inlier_ratio = (result.pass_count > 0) ? result.total_inlier_ratio / result.pass_count : 0.0f;

        if (pass_rate > best_pass_rate)
        {
            best_pass_rate = pass_rate;
            best_avg_inlier_ratio = avg_inlier_ratio;
            best_params = key;
        }
        else if (std::abs(pass_rate - best_pass_rate) < 1e-6 && avg_inlier_ratio > best_avg_inlier_ratio)
        {
            best_avg_inlier_ratio = avg_inlier_ratio;
            best_params = key;
        }
    }

    if (best_pass_rate < 0.0f)
    {
        spdlog::warn("No valid results found to analyze.");
    }
    else
    {
        std::cout << "\n\n--- Analysis Complete: Best Parameters Found ---\n";
        std::cout << "Pass Rate:           " << (best_pass_rate * 100.0) << "%\n";
        std::cout << "Avg. Inlier Ratio:   " << best_avg_inlier_ratio << "\n";
        std::cout << "----------------------------------------------\n";
        std::cout << "max_num_features:         (Dynamically calculated: 5 * sqrt(total pixels))\n";
        std::cout << "initial_gauss_blur:       " << best_params.initial_gauss_blur << "\n";
        std::cout << "extract_sift_thresh:      " << best_params.extract_sift_thresh << "\n";
        std::cout << "num_octaves:              " << best_params.num_octaves << "\n";
        std::cout << "find_homography_thresh:   " << best_params.find_homography_thresh << "\n";
        std::cout << "----------------------------------------------\n\n";
    }
}

// --- Main Function ---
int main(int argc, char *argv[])
{
    spdlog::set_level(spdlog::level::debug);
    TunerConfig config;
    parse_args(argc, argv, config);

    if (config.image_dir.empty() || config.output_csv.empty())
    {
        std::cerr << "Usage: " << argv[0] << " --image_dir <path> --output_csv <path> [param_options...]\n";
        std::cerr << "Example param option: --extract_sift_thresh_range <start> <end> <count>\n";
        return 1;
    }

    auto image_pairs = find_image_pairs(config.image_dir);
    if (image_pairs.empty())
    {
        spdlog::error("No image pairs found in directory: {}", config.image_dir);
        return 1;
    }
    spdlog::info("Found {} image pairs to process.", image_pairs.size());

    auto param_sets = generate_parameter_sets(config);
    if (param_sets.empty())
    {
        spdlog::error("No parameter combinations to test.");
        return 1;
    }
    spdlog::info("Generated {} unique parameter sets to test against each pair.", param_sets.size());

    std::ofstream csv_file(config.output_csv);
    csv_file << "image1,image2,passed_check,inlier_ratio,h_scale_x,h_scale_y,h_shear_x,h_shear_y,time_ms,"
             << "max_num_features,initial_gauss_blur,extract_sift_thresh,num_octaves,find_homography_thresh\n";

    for (const auto &pair : image_pairs)
    {
        spdlog::info("Processing pair: {} and {}", pair.first, pair.second);
        float *h_img1 = nullptr, *h_img2 = nullptr;
        int w1, h1, w2, h2;
        if (!load_and_prepare_image(pair.first, h_img1, w1, h1) || !load_and_prepare_image(pair.second, h_img2, w2, h2))
        {
            continue;
        }

        long long total_pixels = (long long)w1 * h1 + (long long)w2 * h2;
        int dynamic_max_features = static_cast<int>(5.0 * std::sqrt(total_pixels));
        spdlog::info("Dynamic max_num_features for this pair: {}", dynamic_max_features);

        for (const auto &params_base : param_sets)
        {
            cudasift_settings current_params = params_base;
            spdlog::debug("Testing with sift_thresh={}", current_params.extract_sift_thresh);
            auto start_time = std::chrono::high_resolution_clock::now();
            float homography[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float inlier_ratio = 0.0f;
            int result_code = CUDASIFT(h_img1, w1, h1, h_img2, w2, h2, homography, &inlier_ratio, &current_params);
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            bool passed = (result_code == CUSIFT_ERROR_NONE) ? CheckHomography(homography, config.shearing_thresh, config.scaling_thresh) : false;

            csv_file << fs::path(pair.first).filename().string() << ","
                     << fs::path(pair.second).filename().string() << ","
                     << (passed ? "true" : "false") << ","
                     << inlier_ratio << ","
                     << homography[0] << "," << homography[4] << ","
                     << homography[1] << "," << homography[3] << ","
                     << duration_ms << ","
                     << current_params.initial_gauss_blur << ","
                     << current_params.extract_sift_thresh << ","
                     << current_params.num_octaves << ","
                     << current_params.find_homography_thresh << "\n";
        }
        delete[] h_img1;
        delete[] h_img2;
    }
    csv_file.close();
    spdlog::info("Parameter tuning complete. Results saved to {}", config.output_csv);
    analyze_results(config.output_csv);
    return 0;
}
