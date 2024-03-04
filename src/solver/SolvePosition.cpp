#include <SolvePosition.hpp>
#include <opencv2/calib3d.hpp>

solvePos::solvePos()
{
    cameraMatrix = (cv::Mat_<double>(3, 3) << 1803.087631049802, 0, 691.2828800166443,
                                            0, 1799.840750940228, 563.9131667034846,
                                            0, 0, 1);
    distCoeffs = (cv::Mat_<double>(1, 5) << -0.1818785319354298, 2.140217991989159, -0.002596255682579479, 
                                        -0.004227944181382182, -16.14656152986067);
    
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

    // 相机到云台的变换矩阵
    rot_camera2PTZ = (cv::Mat_<double>(3, 3) << cos(-PI / 2.0), -sin(-PI / 2.0), 0,
                                            sin(-PI / 2.0), cos(-PI / 2.0), 0,
                                            0, 0, 1);
    trans_camera2PTZ = (cv::Mat_<double>(3, 1) << 10.0, 0, 0);
}

std::vector<float> solvePos::getAngle(const std::vector<cv::Point2f> &image_points, int class_id)
{
    points2d = image_points;
    id = class_id;

    solvePnP();

    Camera2PTZ();

    float x = PTZ_coord.at<float>(0, 0);
    float y = PTZ_coord.at<float>(1, 0);
    float z = PTZ_coord.at<float>(2, 0);

    float pitch = (x / z) * (180 / PI);
    float yaw = (y / z) * (180 / PI);

    std::vector<float> angle = {pitch, yaw};
    return angle;
}

void solvePos::solvePnP()
{
    if (id == 0 || id == 7) // 大装甲板
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
