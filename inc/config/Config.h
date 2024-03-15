#ifndef __CONFIG_H_
#define __CONFIG_H_

#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>

#define SHOW_OUTPUT
#define SHOW_FPS
// #define TIME
// #define SERIAL_DEBUG
#define CPU_PREPROCESS
// #define GPU_PREPROCESS

enum armo_classes   // 装甲板类别
{
    B1,
    B2,
    B3,
    B4,
    B5,
    BO,
    BS,
    R1,
    R2,
    R3,
    R4,
    R5,
    RO,
    RS
};

typedef struct
{
    armo_classes class_id;         // 类别
    float conf;                    // 置信度
    cv::Rect2f bbox;                 // 检测框
    std::vector<cv::Point2f> kpoints; // 关键点
}result;

const std::string config_file_path("/home/supremacy/Desktop/code/project/res/config.yaml");

class config
{
public:
    config();

public:
    // 串口
    std::string serial_dev;
    // 目标筛选
    int enermy;
    // 相机参数
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    // 相机到云台的变换矩阵
    cv::Mat rot_camera2PTZ;
    cv::Mat trans_camera2PTZ;
    // 大装甲板id
    std::vector<int> large_id;
    // 推理
    std::string engineFile_path;
    float conf_thred;
    float nms_thred;
};

#endif
