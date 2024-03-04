#include "GetImg.h"

//initiate camare
GetImg::GetImg()
{
    this->g_i64ColorFilter = GX_COLOR_FILTER_BAYER_BG;  // Color filter of device
    this->hDevice = nullptr;                            // camare device
    this->status = GX_STATUS_SUCCESS;                   // camare status

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

    //打开第一个设备
    this->status = GXOpenDeviceByIndex(1, &this->hDevice);
    if (this->status != GX_STATUS_SUCCESS)
    {
        exit(-1);
    }

    // 使能帧存覆盖
    this->status = GXSetBool(this->hDevice, GX_BOOL_FRAMESTORE_COVER_ACTIVE, true);
    // 使能采集帧率调节模式
    this->status = GXSetEnum(this->hDevice, GX_ENUM_ACQUISITION_FRAME_RATE_MODE ,
                    GX_ACQUISITION_FRAME_RATE_MODE_ON);
    this->status = GXSetInt(hDevice, GX_INT_ACQUISITION_SPEED_LEVEL, 10000);

    this->status = GXStreamOn(hDevice);

    // 调用 GXDQBuf 取一帧图像用于初始化
    this->status = GXDQBuf(this->hDevice, &this->pFrameBuffer, 1000);
    this->g_pRGBImageBuf = nullptr;     // Memory for RAW8toRGB24
    if (this->status == GX_STATUS_SUCCESS)
    {
        if (this->pFrameBuffer->nStatus == GX_FRAME_STATUS_SUCCESS)
        {
            VxInt32 emDXStatus = DX_OK;
            this->width = pFrameBuffer->nWidth;    // frame width and height
            this->height = pFrameBuffer->nHeight;
            g_pRGBImageBuf = new unsigned char[width * height * 3];  // set RGB image size
        }
    }
}

//get one frame gpu image
cv::Mat GetImg::getImg()
{ 
    if (this->status == GX_STATUS_SUCCESS)
    {
        //调用 GXDQBuf 取一帧图像
        this->status = GXDQBuf(this->hDevice, &this->pFrameBuffer, 1000);
        if (this->status == GX_STATUS_SUCCESS)
        {
            if (this->pFrameBuffer->nStatus == GX_FRAME_STATUS_SUCCESS)
            {
                VxInt32 emDXStatus = DX_OK;
                // Convert to the RGB image
                emDXStatus = DxRaw8toRGB24(this->pFrameBuffer->pImgBuf, this->g_pRGBImageBuf, this->width, this->height,
                                RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(g_i64ColorFilter), false);
                if (emDXStatus != DX_OK)    // Convertion is success
                {
                    this->err = std::string("DxRaw8toRGB24 Failed, Error Code: ");
                    std::cout << this->err << emDXStatus << std::endl;
                    exit(-1);
                }
            }
            else 
            {
                this->err = std::string("pFrameBuffer Error!");
                std::cout << this->err << std::endl;
                exit(-1);
            }
            // 图像拷贝至Frame中
            cv::Mat Frame(cv::Size(this->width, this->height), CV_8UC3, this->g_pRGBImageBuf);
            // 调用 GXQBuf 将图像 buf 放回库中继续采图
            this->status = GXQBuf(this->hDevice, this->pFrameBuffer);
            return Frame;
        }  
        else
        {
            this->err = std::string("GXDQBuf Error!");
            std::cout << this->err << std::endl;
            exit(-1);
        } 
    } 
    else
    {
        this->err = std::string("GXStreamOn Error!");
        std::cout << this->err << std::endl;
        exit(-1);
    }
}

//stop camare stream
GetImg::~GetImg()
{
    delete this->g_pRGBImageBuf;
    //停采
    this->status = GXStreamOff(hDevice);
    this->status = GXCloseDevice(hDevice); 
    this->status = GXCloseLib();
}