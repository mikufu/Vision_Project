#ifndef __PROCIMG_H_
#define __PROCIMG_H_

// using this class to process image
class ProcImg
{
private:
    cv::Mat display;    //create one frame to display

    CFps GF;            //Create a CFps class

public:
    void preProc(cv::cuda::GpuMat& gpuFrame);   // preprocess image

    void showImg(cv::cuda::GpuMat& gpuFrame);   // show result

    void showFps();                             // show Fps

};

#endif
