#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <Fps.h>
#include "ProcImg.h"

void ProcImg::preProc(cv::cuda::GpuMat& gpuFrame)
{
    using namespace cv;

    /*preProcess image using gpu*/
    cuda::resize(gpuFrame, gpuFrame, Size(640, 480));
}

void ProcImg::showImg(cv::cuda::GpuMat& gpuFrame)
{
    using namespace std;
    using namespace cv;

    //gpuFrame download to cpu from gpu
    gpuFrame.download(this->display);

    // show image
    namedWindow("procImg", WINDOW_NORMAL);
    imshow("procImg", this->display);
}

void ProcImg::showFps()
{
    // show display Fps
    this->GF.IncreaseFrameNum();
    this->GF.UpdateFps();
    std::cout << "current Fps: " << GF.GetFps() << std::endl;
}