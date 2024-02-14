#ifndef __PROCIMG_H_
#define __PROCIMG_H_

#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "GetImg.h"
#include "inference.h"

// using this class to process image
class ProcImg
{
private:
    cv::Mat image;  // 原图
    cv::cuda::GpuMat gpu_image;
    cv::Mat output; // 结果图
    cv::cuda::GpuMat gpu_output;;

    GetImg GI;      // 获得图片的类

    int step;       // 用于计算FPS
    std::chrono::_V2::steady_clock::time_point start;
    std::chrono::_V2::steady_clock::time_point end;
    std::string FPS;

    inference *infer;    // 用于推理

    cv::cuda::Stream stream;    // cuda流

public:
    ProcImg();

    ~ProcImg();

    void process();   // preprocess image
private:

    void show();   // show result
};

#endif
