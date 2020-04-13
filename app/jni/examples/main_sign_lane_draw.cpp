#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <assert.h>
#include <dirent.h>
#include <json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "sign_detection.hpp"
#include "lane_detection.hpp"

// using json = nlohmann::json;
static auto LOG = spdlog::stdout_color_mt("MAIN");

void saveSignResult(cv::Mat &im, std::vector<std::vector<ScoreVertices>> result, std::string dstpath)
{
    const int numClass = result.size();
    for (int clsIdx = 0; clsIdx < numClass - 1; ++clsIdx)
    {
        std::vector<ScoreVertices> &signobjvector = result[clsIdx];
        const int numSign = signobjvector.size();
        for (int signIdx = 0; signIdx < numSign; ++signIdx)
        {
            cv::Point vertices[1][4];
            vertices[0][0] = cv::Point(signobjvector[signIdx].x0 * im.cols, signobjvector[signIdx].y0 * im.rows);
            vertices[0][1] = cv::Point(signobjvector[signIdx].x1 * im.cols, signobjvector[signIdx].y1 * im.rows);
            vertices[0][2] = cv::Point(signobjvector[signIdx].x2 * im.cols, signobjvector[signIdx].y2 * im.rows);
            vertices[0][3] = cv::Point(signobjvector[signIdx].x3 * im.cols, signobjvector[signIdx].y3 * im.rows);
            const cv::Point* vtsptr[1] = {vertices[0]};
            int npt[] = {4};
            cv::polylines(im, vtsptr, npt, 1, 1, cv::Scalar(255, 0, 0), 2, 16);
        }
    }
    cv::imwrite(dstpath, im);
}

void saveLaneResult(cv::Mat &im, float *result, const int hpad, const float scoreTh)
{
    const int PLANESIZE = im.cols * (im.rows - hpad);
    uint8_t *shiftsrc = im.data + hpad * im.cols * 3;
    for (int i = 0; i < PLANESIZE; ++i)
    {
        if (result[i] > scoreTh)
        {
            shiftsrc[i * 3 + 1] = 255;
        }
    }
    // cv::imwrite(dstpath, im);
}

int main(int ac, char *av[])
{
    if (ac != 2)
    {
        SLOG_ERROR << av[0] << " [config_file.json]" << std::endl;
        return 1;
    }

    nlohmann::json config;
    std::ifstream fin;
    fin.open(av[1]);
    fin >> config;
    fin.close();


    nlohmann::json signconfig = config["sign"];
    nlohmann::json laneconfig = config["lane"];
    SignDet signdet(signconfig);
    LaneDet lanedet(laneconfig);

    const std::string folderpath = signconfig["evaluate"]["folder_path"].get<std::string>();
    const int NETW = signconfig["model"]["net_width"].get<int>();
    const int NETH = signconfig["model"]["net_height"].get<int>();
    const int PAD = laneconfig["model"]["net_height_cut"].get<int>();
    const float TH = laneconfig["model"]["score_threshold"].get<float>();

    std::string resultpath = folderpath + "result/";
    mkdir(resultpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    DIR *dir;
    struct dirent *entry;
    if (!(dir = opendir(folderpath.c_str())))
    {
        SLOG_ERROR << "Assigned folder path does not exist" << std::endl;
        return 1;
    }
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type != DT_DIR)
        {
            std::string filepath = std::string(entry->d_name);
            std::string ext = std::string(".jpg");
            if ((filepath.length() > ext.length()) && (filepath.compare(filepath.length() - ext.length(), ext.length(), ext) == 0))
            {
                std::string srcfullpath = folderpath + filepath;
                std::string dstfullpath = resultpath + filepath;
                SLOG_INFO << srcfullpath << std::endl;

                cv::Mat im = cv::imread(srcfullpath);
                cv::Mat imnet;
                cv::resize(im(cv::Rect(0, 0, 1280, 768)), imnet, cv::Size(NETW, NETH)); // maybe its no cool to hard code here


                std::vector<std::vector<ScoreVertices>> signresult = signdet.run(imnet.data);
                float *laneresult = lanedet.run(imnet.data);

                saveLaneResult(imnet, laneresult, PAD, TH);
                saveSignResult(imnet, signresult, dstfullpath);
            }
        }
    }


    return 0;
}