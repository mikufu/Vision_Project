#include "ProcImg.h"
#include <opencv2/opencv.hpp>

const int Buffer_Size = 1;
// 每个线程所需要的缓存
std::queue<cv::Mat> t1_frames;
std::queue<void *> t2_buffers;
std::queue<std::vector<result>> t3_item;
std::queue<cv::Mat> t_show;
// 每个线程的互斥锁
std::mutex t1_mutex;
std::mutex t2_mutex;
std::mutex t3_mutex;
// 每个线程的not_full变量
std::condition_variable t1_not_full;
std::condition_variable t2_not_full;
std::condition_variable t3_not_full;
// 每个线程的not_empty变量
std::condition_variable t1_not_empty;
std::condition_variable t2_not_empty;
std::condition_variable t3_not_empty;

ProcImg::ProcImg()
{
    config cfg;
    bool is_succeed = false;

    // 初始化相机
    DH = new DHCam();
    is_succeed = DH->Init();
    CHECK_SUCCEED(is_succeed, "DH->Init()");

    // 初始化串口
    std::cout << cfg.serial_dev << std::endl;
    ser = new serial_port(cfg.serial_dev);
    is_succeed = ser->Init();
    CHECK_SUCCEED(is_succeed, "ser.Init()");

    // 初始化目标筛选类
    targ = new target(cfg.enermy);

    // 初始化坐标解算类
    sol = new solvePos(cfg.cameraMatrix, cfg.distCoeffs, cfg.rot_camera2PTZ, cfg.trans_camera2PTZ, cfg.large_id);
    sol->Init();

    // 初始化推理类
    this->infer = new inference(cfg.engineFile_path, DH->get_width(), DH->get_height(), cfg.conf_thred, cfg.nms_thred);
    infer->Init();
    
    // 开始计时
    this->start = std::chrono::high_resolution_clock::now();
}

void ProcImg::readFrame()
{
    cv::Mat frame;
    while (true)
    {
#ifdef TIME
    COST_TIME(  
#endif

        bool is_successed = DH->getImg(frame);
        if (is_successed == false)
        {
            std::cout << "获取图像失败" << std::endl;
        }

#ifdef TIME
        , "get image"
    );
#endif

        // 互斥锁
        std::unique_lock<std::mutex> lock(t1_mutex);
        // 如果缓存满了就等待
        t1_not_full.wait(lock, []{
            return t1_frames.size() < Buffer_Size;
            });
        // 增加一个元素
        t1_frames.push(frame);
        // 通知下一线程开始
        t1_not_empty.notify_one();
    }
}

void ProcImg::predictFrame()
{
    cv::Mat frame;
    cv::Mat mfloat;
    float *output_data;
    while (true)
    {
       // 使用{} 限制作用域，否则锁会在一次循环结束后才释放
        {
            // 互斥锁
            std::unique_lock<std::mutex> lock(t1_mutex);
            // 如果缓存为空，就等待
            t1_not_empty.wait(lock, []{
                return !t1_frames.empty();
            });
            // 取出一个元素
            frame = t1_frames.front();
            t1_frames.pop();
            // 通知上一线程开始
            t1_not_full.notify_one();
        }

#ifdef SHOW_OUTPUT  // 深拷贝一帧作为结果图
        cv::Mat output_img;
        frame.copyTo(output_img);
        t_show.push(output_img);
#endif

#ifdef TIME
    COST_TIME(
#endif

#ifdef CPU_PREPROCESS   // 预处理
        mfloat = this->infer->CPU_preprocess(frame);
#else
        mfloat = this->infer->GPU_preprocess(frame);
#endif

#ifdef TIME
        , "preprocess"
    );
#endif

        {
            // 互斥锁
            std::unique_lock<std::mutex> lock2(t2_mutex);
            // 如果缓存满了就等待
            t2_not_full.wait(lock2, []{
                return t2_buffers.size() < Buffer_Size;
            });
            // 增加一个元素
            void *input_data = mfloat.ptr<void>(0);
            t2_buffers.push(input_data);
            // 通知下一线程开始
            t2_not_empty.notify_one();
        }
    }
}

void ProcImg::getResult()
{
    void *input_data;
    float *output_data;
    std::vector<result> res;
    double time = 0;
    while (true)
    {
        // 使用{} 限制作用域，否则锁会在一次循环结束后才释放
        {
            // 互斥锁
            std::unique_lock<std::mutex> lock(t2_mutex);
            // 如果缓存为空，就等待
            t2_not_empty.wait(lock, []{
                return !t2_buffers.empty();
            });
            // 取出一个元素
            input_data = t2_buffers.front();
            t2_buffers.pop();
            // 通知上一线程开始
            t2_not_full.notify_one();
        }

#ifdef TIME
    COST_TIME(
#endif

#ifdef TIME
    COST_TIME(
#endif

        output_data = this->infer->predict(input_data); // 预测

#ifdef TIME
        , "predict"
    );
#endif

        this->infer->postprocess(output_data);
        res = this->infer->get_results();
        delete output_data;

        bool is_find = targ->select(res);
        // std::cout << "is_find : " << is_find << std::endl;
        Serial_Data sd;
        if (is_find)
        {
            auto t = targ->get_target();
            // std::cout << "target : " << t.class_id << std::endl;
            std::vector<float> angle = sol->getAngle(t.kpoints, t.class_id);
            float dist = sol->get_distance();
            // std::cout << "angle : " << "yaw = " << angle[0] << " pitch = " << angle[1] << std::endl;

            sd.is_find = 1;
            sd.yaw.f = angle[0];
            sd.pitch.f = angle[1];
            sd.dist.f = dist;
            // std::cout << "distance : " << dist << std::endl;
            DH->setExposureAndGain(is_find, dist);
        }
        else
        {
            sd.is_find = 0;
            DH->setExposureAndGain();
        }
        ser->sendData(sd);

        this->end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end - this->start);
        auto cost = duration.count();

#ifdef SHOW_FPS
        time += cost;
        step++;
        if (time >= 1E6)
        {
            this->FPS = "FPS : " + std::to_string(1.0E6 * step / time).substr(0, 5);
            std::cout << FPS << std::endl;
            step = 0;
            time = 0;
        }
        // std::cout << "total cost time = " << cost << "us" << std::endl;
#endif

        this->start = this->end;

#ifdef TIME
        , "get result"
    );
#endif

#ifdef SHOW_OUTPUT
        {
            // 互斥锁
            std::unique_lock<std::mutex> lock2(t3_mutex);
            // 如果缓存满了就等待
            t3_not_full.wait(lock2, []{
                return t3_item.size() < Buffer_Size;
            });
            // 增加一个元素
            t3_item.push(res);
            // 通知下一线程开始
            t3_not_empty.notify_one();
        }
#endif
    }
}

void ProcImg::kalman()
{

}

void ProcImg::show()
{
    cv::Mat output_img;
    std::vector<result> res;
    while (true)
    {
        {
            // 互斥锁
            std::unique_lock<std::mutex> lock(t3_mutex);
            // 如果缓存为空，就等待
            t3_not_empty.wait(lock, []{
                return !t3_item.empty();
            });
            // 取出一个元素
            res = t3_item.front();
            t3_item.pop();
            output_img = t_show.front();
            t_show.pop();
            // 通知上一线程开始
            t3_not_full.notify_one();
        }

#ifdef TIME
    COST_TIME(
#endif

        this->infer->drawplot(output_img, res);

        if (!output_img.empty())
        {
            cv::putText(output_img, this->FPS, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 200, 0), 3);
            imshow("output_img", output_img);
        }
        char c = cv::waitKey(1);
        if (c == 27 || c == 'q' || c == 'Q')
        {
            std::cout << "线程4退出" << std::endl;
            this->quit = true;
            exit(0);
        }

#ifdef TIME
        , "show"
    );
#endif
    }
}

ProcImg::~ProcImg()
{
    delete DH;
    delete ser;
    delete targ;
    delete sol;
    delete infer;
}
