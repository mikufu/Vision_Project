#include <SolvePosition.hpp>
#include <opencv2/calib3d.hpp>

void solvePos::Init()
{
    // 装甲板三维坐标，以装甲板为OXY平面，装甲板中心为(0,0,0)
    double L_half_x = L_armo_width / 2.0;
    double S_half_x = S_armo_width / 2.0;
    double half_y = light_length / 2.0;
    // 大装甲板的四个点：依次为左上，左下，右下，右上
    L_pos.emplace_back(cv::Point3f(-L_half_x, -half_y, 0));
    L_pos.emplace_back(cv::Point3f(-L_half_x, half_y, 0));
    L_pos.emplace_back(cv::Point3f(L_half_x, half_y, 0));
    L_pos.emplace_back(cv::Point3f(L_half_x, -half_y, 0));
    // 小装甲板的四个点：依次为左上，左下，右下，右上
    S_pos.emplace_back(cv::Point3f(-S_half_x, -half_y, 0));
    S_pos.emplace_back(cv::Point3f(-S_half_x, half_y, 0));
    S_pos.emplace_back(cv::Point3f(S_half_x, half_y, 0));
    S_pos.emplace_back(cv::Point3f(S_half_x, -half_y, 0));
};

std::vector<float> solvePos::getAngle(const std::vector<cv::Point2f> &image_points, int class_id)
{
    points2d = image_points;
    id = class_id;

    solvePnP();

    Camera2PTZ();

    double x = PTZ_coord.at<double>(0, 0);
    double y = PTZ_coord.at<double>(1, 0);
    double z = PTZ_coord.at<double>(2, 0);

    float yaw = atan(x / z) * (180 / PI);
    float pitch = atan(y / z) * (180 / PI);

    std::vector<float> angle = {yaw, pitch};
    return angle;
}

void solvePos::solvePnP()
{
    bool is_large = false;
    for (int i = 0; i < large_id.size(); i++)
    {
        if (id == large_id[i])
        {
            is_large = true;
            break;
        }
    }

    if (is_large) // 大装甲板
    {
        points3d = L_pos;
    }
    else
    {
        points3d = S_pos;
    }
    cv::solvePnP(points3d, points2d, cameraMatrix, distCoeffs, rot, trans);
}

void solvePos::Camera2PTZ()
{
    // 以云台为中心，x轴向上，y轴向右的右手系
    PTZ_coord = rot_camera2PTZ * trans + trans_camera2PTZ;
}
