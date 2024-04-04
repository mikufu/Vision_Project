#include "DHCamera.hpp"

// initiate camare
bool DHCam::Init()
{
    g_i64ColorFilter = GX_COLOR_FILTER_BAYER_BG;  // Color filter of device
    hDevice = nullptr;                            // camare device
    status = GX_STATUS_SUCCESS;                   // camare status

    //init camera device
    uint32_t nDeviceNum = 0;
    //初始化库
    status = GXInitLib();
    GX_VERIFY(status, "GXInitLib()");

    //枚举设备列表
    status = GXUpdateDeviceList(&nDeviceNum, 1000);
    GX_VERIFY(status, "GXUpdateDeviceList(&nDeviceNum, 1000)");

    //打开第一个设备
    status = GXOpenDeviceByIndex(1, &hDevice);
    GX_VERIFY(status, "GXOpenDeviceByIndex(1, &hDevice)");

    // // 使能帧存覆盖
    // status = GXSetBool(hDevice, GX_BOOL_FRAMESTORE_COVER_ACTIVE, true);
    // GX_VERIFY(status, "GXSetBool(hDevice, GX_BOOL_FRAMESTORE_COVER_ACTIVE, true)");

    // 使能采集帧率调节模式
    status = GXSetEnum(hDevice, GX_ENUM_ACQUISITION_FRAME_RATE_MODE ,
                    GX_ACQUISITION_FRAME_RATE_MODE_ON);
    GX_VERIFY(status, "GXSetEnum(hDevice, GX_ENUM_ACQUISITION_FRAME_RATE_MODE , \
                    GX_ACQUISITION_FRAME_RATE_MODE_ON)");

    // status = GXSetInt(hDevice, GX_INT_ACQUISITION_SPEED_LEVEL, 10000);
    // GX_VERIFY(status, "GXSetInt(hDevice, GX_INT_ACQUISITION_SPEED_LEVEL, 10000)");

    // 设置曝光时间
    status = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, dExposureTime);
    GX_VERIFY(status, "GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, dExposureTime)");

    // 设置增益
    status = GXSetFloat(hDevice, GX_FLOAT_GAIN, dGain);
    GX_VERIFY(status, "GXSetFloat(hDevice, GX_FLOAT_GAIN, dGain)");

    // 开启白平衡模式
    status = GXSetEnum(hDevice, GX_ENUM_BALANCE_WHITE_AUTO, GX_BALANCE_WHITE_AUTO_CONTINUOUS);
    GX_VERIFY(status, "GXSetEnum(hDevice, GX_ENUM_BALANCE_WHITE_AUTO, GX_BALANCE_WHITE_AUTO_CONTINUOUS)");

    // 开启采集流
    status = GXStreamOn(hDevice);
    GX_VERIFY(status, "GXStreamOn(hDevice)");

    // 调用 GXDQBuf 取一帧图像用于初始化
    status = GXDQBuf(hDevice, &pFrameBuffer, 1000);
    GX_VERIFY(status, "GXDQBuf(hDevice, &pFrameBuffer, 1000)");

    g_pRGBImageBuf = nullptr;     // Memory for RAW8toRGB24

    if (pFrameBuffer->nStatus == GX_FRAME_STATUS_SUCCESS)
    {
        VxInt32 emDXStatus = DX_OK;
        width = pFrameBuffer->nWidth;    // frame width and height
        height = pFrameBuffer->nHeight;
        g_pRGBImageBuf = new unsigned char[width * height * 3];  // set RGB image size
    }

    return true;
}

// get one frame gpu image
bool DHCam::getImg(cv::Mat &frame)
{ 
    //调用 GXDQBuf 取一帧图像
    status = GXDQBuf(hDevice, &pFrameBuffer, 1000);
    GX_VERIFY(status, "GXDQBuf(hDevice, &pFrameBuffer, 1000)");

    if (pFrameBuffer->nStatus == GX_FRAME_STATUS_SUCCESS)
    {
        VxInt32 emDXStatus = DX_OK;
        // Convert to the RGB image
        emDXStatus = DxRaw8toRGB24(pFrameBuffer->pImgBuf, g_pRGBImageBuf, width, height,
                        RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(g_i64ColorFilter), false);
        if (emDXStatus != DX_OK)    // Convertion is success
        {
            err = std::string("DxRaw8toRGB24 Failed, Error Code: ");
            std::cout << err << emDXStatus << std::endl;
            return false;
        }
    }
    else 
    {
        err = std::string("pFrameBuffer Error!");
        std::cout << err << std::endl;
        return false;
    }
    // 图像拷贝至Frame中
    frame = cv::Mat(cv::Size(width, height), CV_8UC3, g_pRGBImageBuf);
    // 调用 GXQBuf 将图像 buf 放回库中继续采图
    status = GXQBuf(hDevice, pFrameBuffer);

    return true;
}

// 动态设置曝光和增益
bool DHCam::setExposureAndGain(bool is_find, double distance)
{
    GX_STATUS emStatus;
    if (is_find)
    {
        if (distance >= 400.0)
        {
            if ( dExposureTime != 10000.0 && ++frame_cnt % 100 == 0)
            {
                dExposureTime = 10000.0;
                dGain = 12.0;
                emStatus = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, dExposureTime);
                emStatus = GXSetFloat(hDevice, GX_FLOAT_GAIN, dGain);
            }
        }
        else
        {
            if (dExposureTime == 10000.0 && ++frame_cnt % 10 == 0)
            {
                dExposureTime = 6666.6;
                dGain = 18.0;
                emStatus = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, dExposureTime);
                emStatus = GXSetFloat(hDevice, GX_FLOAT_GAIN, dGain);
            }
        }
    }
    else
    {
        if (dExposureTime != 10000.00 && ++frame_cnt % 100 == 0)
        {
            dExposureTime = 10000.0;
            dGain = 12.0;
            emStatus = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, dExposureTime);
            emStatus = GXSetFloat(hDevice, GX_FLOAT_GAIN, dGain);
        }
    }
    GX_VERIFY(status, "设置曝光时间");

    return true;
}

// stop camare stream
DHCam::~DHCam()
{
    delete g_pRGBImageBuf;
    //停采
    status = GXStreamOff(hDevice);
    status = GXCloseDevice(hDevice); 
    status = GXCloseLib();
}
