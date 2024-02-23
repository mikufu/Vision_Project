#ifndef __PROCIMG_H_
#define __PROCIMG_H_

#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "GetImg.h"
#include "inference.h"

#define SHOW_OUTPUT
#define SHOW_FPS
// #define TIME

// using this class to process image
class ProcImg
{
private:
    GetImg GI;      // 获得图片的类

    int step;       // 用于计算FPS
    std::chrono::_V2::steady_clock::time_point start;
    std::chrono::_V2::steady_clock::time_point end;
    std::string FPS;

    inference *infer;    // 用于推理

    bool quit;  // 判断是否退出

public:
    ProcImg();

    ~ProcImg();

    void readFrame();  // 用于采集图片和处理

    void predictFrame();   // 用于推理

    void getResult();   // 获得结果

    void show();    // 显示结果
};

#endif
