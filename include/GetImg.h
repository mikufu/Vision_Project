#ifndef __GETIMG_H_
#define __GETIMG_H_

#define PIXFMT_CVT_FAIL         false              ///< PixelFormatConvert fail
#define PIXFMT_CVT_SUCCESS      true               ///< PixelFormatConvert success

// get image from camera
class GetImg
{
public:
    int64_t g_i64ColorFilter;           //< Color filter of device

    GX_DEV_HANDLE hDevice;              //camare device

    GX_STATUS status;                   //camare status

public:
    //init camera
    GetImg();

    //get one img and transform to Mat
    void getImg(cv::cuda::GpuMat& gpuFrame);

    //stop camare stream
    ~GetImg();

};

#endif