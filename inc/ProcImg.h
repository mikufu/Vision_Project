#ifndef __PROCIMG_H_
#define __PROCIMG_H_

#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "Config.h"
#include "DHCamera.hpp"
#include "Inference.hpp"
#include "SelectTarget.hpp"
#include "SolvePosition.hpp"
#include "Serial.hpp"

// 计时
#define COST_TIME(fun, name) do{ \
    auto t1 = std::chrono::high_resolution_clock::now(); \
    fun \
    auto t2 = std::chrono::high_resolution_clock::now(); \
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1); \
    std::cout << name << " cost time : " << time.count()  << " us" << std::endl; \
} while(0)

// 检查初始化是否成功
#define CHECK_SUCCEED(is_succeed, func) \
        if (is_succeed != true) \
        { \
            std::cout << func << " error" << std::endl; \
            exit(-1); \
        }

// using this class to process image
class ProcImg
{
private:
    DHCam *DH;      // 获得图片的类
    serial_port *ser; // 用于串口通信
    target *targ;    // 用于筛选装甲板
    solvePos *sol;   // 用于解算坐标
    inference *infer;    // 用于推理

    int step;       // 用于计算FPS
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    std::string FPS;

    bool quit;  // 判断是否退出

public:
    ProcImg();

    ~ProcImg();

    void readFrame();  // 用于采集图片

    void predictFrame();   // 用于推理

    void getResult();   // 获得结果

    void show();    // 显示结果
};

#endif
