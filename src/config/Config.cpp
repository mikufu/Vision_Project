#include "Config.h"

config::config()
{
    cv::FileStorage fs("/home/supremacy/Desktop/code/project/res/config.yaml", cv::FileStorage::READ);

    // 串口设备
    fs["Serial Device"] >> serial_dev;

    fs["Enermy"] >> enermy;

    // 相机参数
    fs["Camera Matrix"] >> cameraMatrix;
    fs["DistCoeffs"] >> distCoeffs;
    // 相机到云台的变换矩阵
    fs["Rotation Matrix Camera To PTZ"] >> rot_camera2PTZ;
    fs["Transform Matrix Camera To PTZ"] >> trans_camera2PTZ;
    // 大装甲板id
    fs["Large Armo ID"] >> large_id;

    // 推理参数
    fs["Engine File Path"] >> engineFile_path;
    fs["Confidence Threshold"] >> conf_thred;
    fs["NMS Threshold"] >> nms_thred;

    fs.release();

}
