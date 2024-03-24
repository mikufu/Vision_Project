#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include "ProcImg.h"

int main()
{
    using namespace cv;

    if (cuda::getCudaEnabledDeviceCount() == 0)
    {
        std::cout << "找不到GPU设备!" << std::endl;
        return -1;
    }

    ProcImg PI;

    std::thread t1(&ProcImg::readFrame, std::ref(PI));   // 获取图像
    std::thread t2(&ProcImg::preprocImage, std::ref(PI));    // 推理
    std::thread t3(&ProcImg::predictFrame, std::ref(PI));  // 获得结果
    std::thread t4(&ProcImg::kalman, std::ref(PI)); // 卡尔曼预测与收发数据

#ifdef SHOW_OUTPUT
    std::thread t_show(&ProcImg::show, std::ref(PI));   // 显示结果
    t_show.join();
#endif

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    return 0;
}