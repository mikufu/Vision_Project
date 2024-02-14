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
    PI.process();

    return 0;
}