#ifndef __INFERENCE_H_
#define __INFERENCE_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "logger.h"

// 输入张量属性
const int input_height = 640;
const int input_width = 640;
const int input_channel = 3;

// yolo输出层
const int rows = 26;
const int cols = 8400;
// 置信度和iou
const float conf_thred = 0.6f;
const float nms_thred = 0.5f;

// 数量
const int nc = 14;
const int nk = 4;

enum armo_classes   // 装甲板类别
{
    B1,
    B2,
    B3,
    B4,
    B5,
    BO,
    BS,
    R1,
    R2,
    R3,
    R4,
    R5,
    RO,
    RS
};

struct result
{
    armo_classes class_id;         // 类别
    float conf;                    // 置信度
    cv::Rect bbox;                 // 检测框
    std::vector<cv::Point> kpoint; // 关键点
};

class inference
{
private:
    // 创建runtime engine context stream
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *mEngine;
    nvinfer1::IExecutionContext *context;
    cudaStream_t infer_stream;
    // 输入层和输出层
    const char *IO_Name[2];
    size_t IO_Size[2];
    // 缩放因子
    float r;
    // 检测结果容器
    std::vector<result> results;
    // 绘制结果
    cv::Mat img_result;
    // 调色盘
    std::vector<cv::Scalar> palette;
    // CUDA memory for input & output
    void *IO_Buf[2];
    // 用于存放推理结果
    float *output_data;

public:
    inference(const int img_height, const int img_width);

    ~inference();

    void predict(cv::Mat img);  // 预测

    std::vector<result> get_results();  // 获得所有结果

    cv::Mat get_img_result();   // 获得绘制结果

private:
    cv::cuda::GpuMat preprocess(cv::Mat img); // 将图片处理成输入格式

    void postprocess();   // 处理输出结果

    void drawplot();    // 绘制结果
};

#endif
