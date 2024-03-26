#include "ProcImg.h"
#include <opencv2/opencv.hpp>

volatile bool is_find;
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
    is_find = false;
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

void ProcImg::preprocImage()
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

void ProcImg::predictFrame()
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

        this->infer->postprocess(output_data);  // 后处理
        res = this->infer->get_results();
        delete output_data;

        while (is_find == true)
            ;
        is_find = targ->select(res);

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
    // std::cout << "is_find : " << is_find << std::endl;
    bool last_find = false;
    unsigned int cnt = 0;
    cv::KalmanFilter kf(6, 3, 0);
    // 状态转移矩阵A
    kf.transitionMatrix = (cv::Mat_<double>(6, 6) << 1, 0, 0, 1, 0, 0,
                                                     0, 1, 0, 0, 1, 0,
                                                     0, 0, 1, 0, 0, 1,
                                                     0, 0, 0, 1, 0, 0,
                                                     0, 0, 0, 0, 1, 0,
                                                     0, 0, 0, 0, 0, 1);
    // 测量矩阵H
    kf.measurementMatrix = (cv::Mat_<double>(3, 6) << 1, 0, 0, 0, 0, 0,
                                                      0, 1, 0, 0, 0, 0,
                                                      0, 0, 1, 0, 0, 0);
    // 系统噪声协方差矩阵Q
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3));
    // 测量噪声协方差矩阵R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));

    while (true)
    {
        // std::cout << "is_find : " << is_find << std::endl;
#ifdef TIME
    COST_TIME(
#endif
        if (is_find == true)
        {
            // std::cout << "kalman1" << std::endl;
            auto t = targ->get_target();
            auto angle = sol->getAngle(t.kpoints, t.class_id);
            float dist = sol->get_distance();
            DH->setExposureAndGain(is_find, dist);  // 根据目标动态调整相机曝光和增益

            Serial_Data sd;
            sd.is_find = true;
            sd.yaw.f = angle[0];
            sd.pitch.f = angle[1];
            sd.dist.f = dist;
            ser->sendData(sd);

            auto PTZ_coord = sol->get_PTZ_coord();
            std::vector<float> PTZ_angle;
            ser->readData(PTZ_angle);
            if (!PTZ_angle.empty())
            {
                float yaw = PTZ_angle[0];
                float pitch = PTZ_angle[1];
                // 将枪口坐标系转换为云台坐标系
                cv::Mat rotate_x = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                                                            0, cos(pitch), sin(pitch),
                                                            0, -sin(pitch), cos(pitch));
                cv::Mat rotate_y = (cv::Mat_<double>(3, 3) << cos(yaw), 0, -sin(yaw),
                                                            0, 1, 0,
                                                            sin(yaw), 0, cos(yaw));
                cv::Mat coord = rotate_x * rotate_y * PTZ_coord;
                // 设置后验状态估计向量和后验估计协方差或更新参数
                if (last_find == true)
                {
                    kf.correct(coord);
                }
                else
                {
                    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
                    kf.statePost = (cv::Mat_<double>(6, 1) << coord.at<double>(0, 0),
                                                              coord.at<double>(1, 0),
                                                              coord.at<double>(2, 0),
                                                              0, 0, 0);
                    last_find = true;
                }
            }
            if (cnt != 0)
            {
                cnt = 0;
            }
        }
        else
        {
            if (last_find == true && ++cnt == MAX_COUNT)
            {
                cnt = 0;
                last_find = false;
                DH->setExposureAndGain();
            }
            Serial_Data sd;
            sd.is_find = 0;
            ser->sendData(sd);
        }
        is_find = false;
        // 卡尔曼预测以及收发数据
        while (is_find == false && last_find == true)
        {
            std::cout << "kalman3" << std::endl;
            cv::Mat coord = kf.predict();
            double x = coord.at<double>(0, 0);
            double y = coord.at<double>(1, 0);
            double z = coord.at<double>(2, 0);
            std::vector<float> PTZ_angle;
            ser->readData(PTZ_angle);
            if (!PTZ_angle.empty())
            {
                Serial_Data sd;
                sd.is_find = 1;
                sd.yaw.f = atan(x / z) - PTZ_angle[0];
                sd.pitch.f = atan(y / z) - PTZ_angle[1];
                sd.dist.f = sqrt(x * x + y * y + z * z);
                ser->sendData(sd);
            }
        }
#ifdef TIME
        , "kalman"
    );
#endif
        usleep(1000);
    }
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
