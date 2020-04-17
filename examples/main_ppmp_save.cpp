#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <future>
#include <sys/stat.h> // mkdir
#include <signal.h>
#include <ppmp/ppmp.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace savevepp
{
    std::promise<void> isExit;
    std::ofstream fout;
    std::string recordPath;

    void sighandler(int signo)
    {
        isExit.set_value();
    }

    void waitExit(void)
    {
        signal(SIGINT, sighandler);
        auto waitExitCmd = isExit.get_future();
        waitExitCmd.wait();
        std::cout << "Done executing the program, shutting down, please wait" << std::endl;
    }

    void openRecord(char *path)
    {
        recordPath = std::string(path);
        mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        fout.open(recordPath + "/vepp6dof.csv", std::ios::out | std::ios::trunc);
        fout << "frameId,timestamp,transX,transY,transZ,rotX,rotY,rotZ,veloX,veloY,veloZ,lat,lon,alt,heading" << std::endl;
    }

    void closeRecord()
    {
        fout.close();
    }

    void camframeCB(const ppmp::CamFrame &camFrame)
    {
        if(camFrame.data)
        {
            cv::Mat imyuv(800, 1280, CV_8UC2, (void *)camFrame.data);
            cv::Mat imbgr;
            cv::cvtColor(imyuv, imbgr, cv::COLOR_YUV2BGR_YUYV);
            cv::imwrite(recordPath + "/" + std::to_string(camFrame.frameId) + ".png", imbgr);
        }
    }

    void positionCB(const ppmp::Pose &pose)
    {
        double sy = std::sqrt(pose.Rec[0][0] * pose.Rec[0][0] + pose.Rec[1][0] * pose.Rec[1][0]);
        bool singular = sy < 1e-6;
        double x_angle_rad = 0.0;
        double y_angle_rad = 0.0;
        double z_angle_rad = 0.0;
        if (!singular)
        {
            x_angle_rad = std::atan2(pose.Rec[2][1] , pose.Rec[2][2]);
            y_angle_rad = std::atan2(-pose.Rec[2][0], sy);
            z_angle_rad = std::atan2(pose.Rec[1][0], pose.Rec[0][0]);
        }
        else
        {
            x_angle_rad = std::atan2(-pose.Rec[1][2], pose.Rec[1][1]);
            y_angle_rad = std::atan2(-pose.Rec[2][0], sy);
            z_angle_rad = 0.0;
        }


        std::stringstream ss;
        ss << std::fixed << pose.frameId << ',' << 
        std::setprecision(6) << pose.timestamp << ',' <<
        pose.Tec[0] << ',' << pose.Tec[1] << ',' << pose.Tec[2] << ',' <<
        x_angle_rad << ',' << y_angle_rad << ',' << z_angle_rad << ',' << 
        pose.Venur[0] << ',' << pose.Venur[1] << ',' << pose.Venur[2] << ',' <<
        std::setprecision(10) << 
        pose.Tllar[0] << ',' << pose.Tllar[1] << ',' << pose.Tllar[2] << ',' << pose.headingr << std::endl;
        fout << ss.str();
    }
}

int main(int ac, char *av[])
{
    if (ac != 2)
    {
        std::cout << av[0] << " [SAVING PATH]" << std::endl;
        return 1;
    }

    savevepp::openRecord(av[1]);

    ppmp::PPMP::Initialize("config/calibration_navinfo_novatel_18Jun2019.xml", "config/sensor_driver.xml");
    ppmp::PPMP::InitializePositioning("config/Navinfo_latest_06182019.pgm");
    ppmp::PPMP::RegisterPositioningOutputCB(savevepp::positionCB);
    ppmp::PPMP::RegisterCamFrameCB(savevepp::camframeCB);
    ppmp::PPMP::Start();

    savevepp::waitExit();
    ppmp::PPMP::Stop();
    savevepp::closeRecord();

    return 0;
}