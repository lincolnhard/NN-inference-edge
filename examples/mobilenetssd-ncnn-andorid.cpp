#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ncnn/platform.h>
#include <ncnn/net.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_mobilenet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net mobilenet;

    mobilenet.load_param("mobilenet_ssd_voc_ncnn.param");
    mobilenet.load_model("mobilenet_ssd_voc_ncnn.bin");

    const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);





    ncnn::Mat out;

    double timesum = 0.0;
    for (int t = 0; t < 100; ++t)
    {
        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

        ncnn::Extractor ex = mobilenet.create_extractor();
        // ex.set_num_threads(4);
        ex.input("data", in);
        ex.extract("detection_out", out);

        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count() << std::endl;
        // timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }

    // std::cout << "MobileNetSSD FPS: " << 1.0 / (timesum / 3000 / 1000.0) << std::endl;

    return 0;
}

int main(int argc, char** argv)
{
    ncnn::Mat in = ncnn::Mat(300, 300, 3);
    in.fill(0.01f);

    return 0;
}

