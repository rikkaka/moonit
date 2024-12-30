#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "moon.h"

#define MOONSIZE 160

std::string moons_txt[] = {"🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"};

void transparent_to_black(cv::Mat &img)
{
    if (img.channels() != 4)
    {
        return;
    }

    std::vector<cv::Mat> chans(4);
    cv::split(img, chans);

    cv::Mat mask;
    cv::compare(chans[3], 0, mask, cv::CMP_EQ);

    chans[0].setTo(0, mask);
    chans[1].setTo(0, mask);
    chans[2].setTo(0, mask);

    cv::merge(chans, img);

    cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
}

float compute_SSIMMap(const cv::Mat &I1, const cv::Mat &I2)
{

    const float C1 = 6.5025f;
    const float C2 = 58.5225f;

    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;
    {
        cv::Mat tmp;
        cv::GaussianBlur(I1_2, tmp, cv::Size(11, 11), 1.5);
        sigma1_2 = tmp - mu1_2;
    }
    {
        cv::Mat tmp;
        cv::GaussianBlur(I2_2, tmp, cv::Size(11, 11), 1.5);
        sigma2_2 = tmp - mu2_2;
    }
    {
        cv::Mat tmp;
        cv::GaussianBlur(I1_I2, tmp, cv::Size(11, 11), 1.5);
        sigma12 = tmp - mu1_mu2;
    }

    cv::Mat t1 = 2.f * mu1_mu2 + C1;
    cv::Mat t2 = 2.f * sigma12 + C2;
    cv::Mat t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssimMap;
    cv::divide(t3, t1, ssimMap);

    auto ssim_channels = cv::mean(ssimMap);
    return ssim_channels[0] * 0.8f + ssim_channels[1] * 0.1f + ssim_channels[2] * 0.1f;
}

cv::Mat *load_moons()
{
    cv::Mat *moons = new cv::Mat[8];
    unsigned char *arr_moon[] = {
        moon0_png, moon1_png, moon2_png, moon3_png,
        moon4_png, moon5_png, moon6_png, moon7_png};
    unsigned int arr_moon_size[] = {
        moon0_png_len, moon1_png_len, moon2_png_len, moon3_png_len,
        moon4_png_len, moon5_png_len, moon6_png_len, moon7_png_len};

    for (int i = 0; i < 8; i++)
    {
        std::vector<unsigned char> data(arr_moon[i], arr_moon[i] + arr_moon_size[i]);
        cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
        cv::resize(img, img, cv::Size(MOONSIZE, MOONSIZE));
        moons[i] = img;
    }
    return moons;
}

void preprocess_moons(cv::Mat *moons, int cell_width, int cell_height)
{
    for (int i = 0; i < 8; i++)
    {

        transparent_to_black(moons[i]);

        cv::cvtColor(moons[i], moons[i], cv::COLOR_BGR2YCrCb);

        if (moons[i].cols != cell_width || moons[i].rows != cell_height)
        {
            cv::resize(moons[i], moons[i], cv::Size(cell_width, cell_height));
        }

        moons[i].convertTo(moons[i], CV_32F);
    }
}

std::string get_most_similar_moon(const cv::Mat *moons, const cv::Mat &cell)
{
    double max_ssim = -1.0;
    int max_index = 0;
    for (int i = 0; i < 8; i++)
    {

        double ssim = compute_SSIMMap(moons[i], cell);
        if (ssim > max_ssim)
        {
            max_ssim = ssim;
            max_index = i;
        }
    }
    return moons_txt[max_index];
}

void enhance_ychannel(cv::Mat &img)
{
    std::vector<cv::Mat> ch(3);
    cv::split(img, ch);

    ch[0].convertTo(ch[0], CV_8U);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(3);
    clahe->setTilesGridSize(cv::Size(16, 16));
    clahe->apply(ch[0], ch[0]);

    cv::merge(ch, img);
}

std::string perform_conversion(const cv::Mat &img, int rows, int cols)
{

    int cell_width = img.cols / cols;
    int cell_height = img.rows / rows;

    cv::Mat *moons = load_moons();

    preprocess_moons(moons, cell_width, cell_height);

    std::string result;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {

            cv::Rect roi(j * cell_width, i * cell_height, cell_width, cell_height);
            cv::Mat cell = img(roi);

            std::string moon = get_most_similar_moon(moons, cell);
            result += moon;
        }
        result += "\n";
    }

    delete[] moons;
    return result;
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " <image> <rows> <columns>\n";
        return -1;
    }

    #if defined(_WIN32) || defined(_WIN64)
        std::system("chcp 65001");
        std::locale::global(std::locale("en_US.UTF-8"));
        std::cout.imbue(std::locale("en_US.UTF-8"));
    #endif

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    cv::Mat input_img = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    if (input_img.empty())
    {
        std::cout << "Could not open or find the image\n";
        return -1;
    }

    transparent_to_black(input_img);

    cv::cvtColor(input_img, input_img, cv::COLOR_BGR2YCrCb);

    int rows = std::stoi(argv[2]);
    int cols = std::stoi(argv[3]);

    cv::resize(input_img, input_img, cv::Size(cols * MOONSIZE, rows * MOONSIZE));

    enhance_ychannel(input_img);

    cv::Mat input_img_float;
    input_img.convertTo(input_img_float, CV_32F);

    std::string result = perform_conversion(input_img_float, rows, cols);

    std::cout << result;

    return 0;
}
