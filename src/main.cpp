#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "GxIAPI.h"
#include "DxImageProc.h"
#include "GetImg.h"
#include "Fps.h"
#include "ProcImg.h"

int main()
{
    using namespace cv;

    //create a GetImg class
    GetImg GI;
    //create a ProcImg class
    ProcImg PI;

    cuda::GpuMat gpuFrame;  //create one gpu frame
    
    //开始循环采集图像
    while(true)
    {
        // get one raw frame
         GI.getImg(gpuFrame);

        // preprocess image
        PI.preProc(gpuFrame);

        // show image
        PI.showImg(gpuFrame);
        // show fps
        PI.showFps();

        char c = waitKey(1);
        if (c == 27 || c == 'q')
            break;
   }

    return 0;
}