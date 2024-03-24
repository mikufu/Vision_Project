#ifndef __GETIMG_H_
#define __GETIMG_H_

#include <iostream>
#include <opencv2/core/mat.hpp>
#include "GxIAPI.h"
#include "DxImageProc.h"

#define PIXFMT_CVT_FAIL         false              ///< PixelFormatConvert fail
#define PIXFMT_CVT_SUCCESS      true               ///< PixelFormatConvert success

/// Judging current error ,show Error message and exit
#define GX_VERIFY(emStatus, func) \
        if(emStatus != GX_STATUS_SUCCESS) \
        { \
            std::cout << func << " error" << std::endl; \
            return false; \
        }


// get image from camera
class DHCam
{
private:
    int64_t g_i64ColorFilter;       // Color filter of device

    GX_DEV_HANDLE hDevice;          // camare device

    GX_STATUS status;               // camare status

    std::string err;                // 错误信息

    PGX_FRAME_BUFFER pFrameBuffer;  // 定义 GXDQBuf 的传入参数

    unsigned char* g_pRGBImageBuf;  // Memory for RAW8toRGB24

    double dExposureTime;    // 曝光时间 us
    double dGain;    // 增益 db

    int width;                      // 图片宽度
    int height;                     // 图片高度

    unsigned int frame_cnt; // 防止曝光时间频繁切换

public:
    //init camera
    DHCam(const double exposure = 10000.0, const double gain = 12.0) :
        dExposureTime(exposure), 
        dGain(gain)
    {
    };

    // 初始化相机参数
    bool Init();

    //get one img and transform to Mat
    bool getImg(cv::Mat &frame);

    // 动态设置曝光和增益
    bool setExposureAndGain(bool is_find = false, double distance = 0.0);

    //stop camare stream
    ~DHCam();

    const int get_width() const
    {
        return this->width;
    };

    const int get_height() const
    {
        return this->height;
    };

};

#endif