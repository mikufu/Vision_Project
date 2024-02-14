#include "ProcImg.h"

ProcImg::ProcImg()
{
    cv::Mat firstFrame = GI.getImg();
    this->infer = new inference(firstFrame.rows, firstFrame.cols); // 初始化推理类
    cv::namedWindow("image", cv::WINDOW_GUI_NORMAL);    // 显示原图片
    cv::namedWindow("output", cv::WINDOW_GUI_NORMAL);   // 显示结果
    this->step = 0;
    this->start = std::chrono::steady_clock::now();
}

void ProcImg::process()
{

    // 开始循环采集图像
    while(true)
    {
        // get one frame
        this->image = this->GI.getImg();
        // this->image = cv::imread("../res/100.jpg");

        // predict
        this->infer->predict(this->image);

        // show image
        // this->show();

        if (++this->step % 50 == 0)
        {
            this->end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->end - this->start);
            this->FPS = "FPS : " + std::to_string(50000.0 / duration.count()).substr(0, 5);
            this->step = 0;
            this->start = this->end;
            std::cout << FPS << std::endl;
        }

        char c = cv::waitKey(1);
        if (c == 27 || c == 'q')
            break;
   }
}

void ProcImg::show()
{
    using namespace std;
    using namespace cv;

    // show output
    this->output = infer->get_img_result();
    cv::putText(this->image, this->FPS, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 200, 0), 3);
    imshow("image", this->image);
    if (!this->output.empty())
    {
        cv::putText(this->output, this->FPS, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 200, 0), 3);
        imshow("output", this->output);
    }
}

ProcImg::~ProcImg()
{
    delete infer;
}
