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

// using json = nlohmann::json;
static auto LOG = spdlog::stdout_color_mt("MAIN");


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
    SignDet signdet(signconfig);

    const std::string folderpath = signconfig["evaluate"]["folder_path"].get<std::string>();
    const int NETW = signconfig["model"]["net_width"].get<int>();
    const int NETH = signconfig["model"]["net_height"].get<int>();

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
                // cv::Mat imnet;
                // cv::resize(im(cv::Rect(0, 0, 1280, 768)), imnet, cv::Size(NETW, NETH)); // maybe its no cool to hard code here

                // signdet.run(imnet.data, imnet.cols, imnet.rows, dstfullpath);
                signdet.run(im.data, im.cols, im.rows, dstfullpath);
            }
        }
    }


    return 0;
}