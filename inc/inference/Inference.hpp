#ifndef __INFERENCE_H_
#define __INFERENCE_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "logger.h"
#include "Config.h"

#define CPU_PREPROCESS
// #define GPU_PREPROCESS

// 数量
const int nc = 14;  // 类别
const int nk = 4;   // 关键点

class inference
{
private:
    // 模型路径
    std::string engineFile_path;
    // 创建runtime engine context stream
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *mEngine;
    nvinfer1::IExecutionContext *context;
    cudaStream_t ToD_stream;
    cudaStream_t infer_stream;
    cudaStream_t ToH_stream;
    // 输入层和输出层
    const char *IO_Name[2];
    size_t IO_Size[2];
    // 缩放因子
    float r;
    // 检测结果容器
    std::vector<result> results;
    // 调色盘
    std::vector<cv::Scalar> palette;
    // CUDA memory for input & output
    void *IO_Buf[2];
    void *IO_Buf2[2];
    // 处理图片的W、H
    int img_width;
    int img_height;
    // 输入张量属性
    int input_channel;
    int input_height;
    int input_width;
    // yolo输出层
    int rows;
    int cols;
    // 置信度和iou
    float conf_thred;
    float nms_thred;

public:
    inference(const std::string engineFile_path, const int width, const int height, 
                const float conf_thred = 0.3, const float nms_thred = 0.5) :
        engineFile_path(engineFile_path),
        img_width(width), 
        img_height(height),
        conf_thred(conf_thred),
        nms_thred(nms_thred)
    {
    };

    ~inference();

    void Init();

    float *predict(void *input_data);  // 预测

    cv::cuda::GpuMat GPU_preprocess(cv::Mat img); // 将图片处理成NCWH格式

    cv::Mat CPU_preprocess(cv::Mat img);

    void postprocess(float *output_data);   // 处理输出结果

    void drawplot(cv::Mat &output, std::vector<result> res);    // 绘制结果
    
    const std::vector<result> get_results() const     // 获得所有结果
    {
        return this->results;
    };
};

#endif
