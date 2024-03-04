#ifndef __GETIMG_H_
#define __GETIMG_H_

#define PIXFMT_CVT_FAIL         false              ///< PixelFormatConvert fail
#define PIXFMT_CVT_SUCCESS      true               ///< PixelFormatConvert success

#include <iostream>
#include <opencv2/opencv.hpp>
#include "GxIAPI.h"
#include "DxImageProc.h"

// get image from camera
class GetImg
{
private:
    int64_t g_i64ColorFilter;       // Color filter of device

    GX_DEV_HANDLE hDevice;          // camare device

    GX_STATUS status;               // camare status

    std::string err;                // 错误信息

    PGX_FRAME_BUFFER pFrameBuffer;  // 定义 GXDQBuf 的传入参数

    unsigned char* g_pRGBImageBuf;  // Memory for RAW8toRGB24

    int width;                      // 图片宽度
    int height;                     // 图片高度

public:
    //init camera
    GetImg();

    //get one img and transform to Mat
    cv::Mat getImg();

    //stop camare stream
    ~GetImg();

};

#endif