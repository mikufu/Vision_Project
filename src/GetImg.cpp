#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "GxIAPI.h"
#include "DxImageProc.h"
#include "GetImg.h"

//initiate camare
GetImg::GetImg()
{
    this->g_i64ColorFilter = GX_COLOR_FILTER_BAYER_BG;              // Color filter of device
    this->hDevice = nullptr;                                        // camare device
    this->status = GX_STATUS_SUCCESS;                               // camare status


    //init camera device
    uint32_t nDeviceNum = 0;
    //初始化库
    this->status = GXInitLib();
    if (this->status != GX_STATUS_SUCCESS)
    {
        exit(-1);
    }
    //枚举设备列表
    this->status = GXUpdateDeviceList(&nDeviceNum, 1000);
    if ((this->status != GX_STATUS_SUCCESS) || (nDeviceNum <= 0))
    {
        exit(-1);
    }
    // cout << "Open success!"<<endl;

    //打开第一个设备
    this->status = GXOpenDeviceByIndex(1, &this->hDevice);
    if (this->status != GX_STATUS_SUCCESS)
    {
        exit(-1);
    }

    // 使能帧存覆盖
    status = GXSetBool(this->hDevice, GX_BOOL_FRAMESTORE_COVER_ACTIVE, true);
    // 使能采集帧率调节模式
    status = GXSetEnum(this->hDevice, GX_ENUM_ACQUISITION_FRAME_RATE_MODE ,
                    GX_ACQUISITION_FRAME_RATE_MODE_ON);
    this->status = GXSetInt(hDevice, GX_INT_ACQUISITION_SPEED_LEVEL, 1000000000);

    this->status = GXStreamOn(hDevice);
}

//get one frame gpu image
void GetImg::getImg(cv::cuda::GpuMat& gpuFrame)
{
    std::string err;                //错误信息 
    //定义 GXDQBuf 的传入参数
    PGX_FRAME_BUFFER pFrameBuffer;
    if (this->status == GX_STATUS_SUCCESS)
    {
        //调用 GXDQBuf 取一帧图像
        status = GXDQBuf(hDevice, &pFrameBuffer, 1000);
        unsigned char* g_pRGBImageBuf = nullptr;     // Memory for RAW8toRGB24
        if (status == GX_STATUS_SUCCESS)
        {
            if (pFrameBuffer->nStatus == GX_FRAME_STATUS_SUCCESS)
            {
                VxInt32 emDXStatus = DX_OK;
                int width = pFrameBuffer->nWidth;    // frame width and height
                int height = pFrameBuffer->nHeight;
                g_pRGBImageBuf = new unsigned char[width * height * 3];  // set RGB image size
                // std::cout << width << '*' <<  height << std::endl;

                // Convert to the RGB image
                emDXStatus = DxRaw8toRGB24(pFrameBuffer->pImgBuf, g_pRGBImageBuf, width, height,
                                RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(g_i64ColorFilter), false);
                if (emDXStatus != DX_OK)    // Convertion is success
                {
                    err = std::string("DxRaw8toRGB24 Failed, Error Code: ");
                    std::cout << err << emDXStatus << std::endl;
                    exit(-1);
                }
                
                cv::Mat Frame(cv::Size(width, height), CV_8UC3, g_pRGBImageBuf);
                gpuFrame.upload(Frame);    //GpuMat格式便于处理和显示

                // release memory
                delete[] g_pRGBImageBuf;
                g_pRGBImageBuf = nullptr;
            }
            else 
            {
                err = std::string("pFrameBuffer Error!");
                std::cout << err << std::endl;
                exit(-1);
            }
            //调用 GXQBuf 将图像 buf 放回库中继续采图
            status = GXQBuf(hDevice, pFrameBuffer);
        }  
        else
        {
            err = std::string("GXDQBuf Error!");
            std::cout << err << std::endl;
            exit(-1);
        } 
    } 
    else
    {
        err = std::string("GXStreamOn Error!");
        std::cout << err << std::endl;
        exit(-1);
    }
}

//stop camare stream
GetImg::~GetImg()
{
    //停采
    this->status = GXStreamOff(hDevice);
    this->status = GXCloseDevice(hDevice); 
    this->status = GXCloseLib();
}