#pragma once

#include <iostream>
#include <opencv2/core/mat.hpp>

const float light_length = 5.0;
const float L_armo_width = 22.5;
const float S_armo_width = 13.0;
const double PI = 3.141592653589793;

class solvePos
{
private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Point3f> L_pos; // 大装甲板三维坐标
    std::vector<cv::Point3f> S_pos; // 小装甲板三维坐标
    std::vector<cv::Point2f> points2d;  // 二维坐标
    std::vector<cv::Point3f> points3d;  // 三维坐标
    int id; // 装甲板类型
    cv::Mat rot;    // 世界坐标系到相机坐标系的旋转矩阵
    cv::Mat trans;  // 位移矩阵
    double dist;    // 距离
    cv::Mat rot_camera2PTZ; // 相机坐标系到云台坐标系的旋转矩阵
    cv::Mat trans_camera2PTZ;   // 相机到云台的位移矩阵
    cv::Mat PTZ_coord;  // 云台坐标
    float pitch;    // 云台pitch轴
    float yaw;  // 云台yaw轴

public:
    solvePos();

    std::vector<float> getAngle(const std::vector<cv::Point2f> &image_points, int class_id);

    cv::Mat get_trans()
    {
        return this->trans;
    }

    double get_distance()
    {
        dist = sqrt(trans.at<double>(0, 0) * trans.at<double>(0, 0) 
                + trans.at<double>(1, 0) * trans.at<double>(1, 0)
                + trans.at<double>(2, 0) * trans.at<double>(2, 0));

        return this->dist;
    }

private:
    void solvePnP();

    void Camera2PTZ();
};