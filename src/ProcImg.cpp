#include "ProcImg.h"
#include <opencv2/opencv.hpp>

const int Buffer_Size = 10;
// 每个线程所需要的缓存
std::queue<cv::Mat> t1_frame;
std::queue<float *> t2_buffers;
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
    bool is_successed = this->sp.init_serial();
    if (is_successed == false)
        exit(-1);
    cv::Mat firstFrame = this->GI.getImg();
    this->infer = new inference(firstFrame.cols, firstFrame.rows); // 初始化推理类
    // this->step = 0;
    this->start = std::chrono::steady_clock::now();
    this->quit = false;
}

void ProcImg::readFrame()
{
    cv::Mat frame;
    while (true)
    {
        // 检查是否退出
        if (this->quit)
        {
            std::cout << "线程1退出" << std::endl;
            break;
        }

#ifdef TIME
        auto t1 = std::chrono::steady_clock::now(); // 计时开始
#endif

        this->GI.getImg().copyTo(frame);
        // cv::Mat adjust;
        // cv::convertScaleAbs(frame, adjust, 2.5, 2.0);

#ifdef TIME
        // 计时结束
        auto t2 = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        if (duration.count() >= 10)
            std::cout << "thread1 cost time : " << duration.count()  << " ms" << std::endl;
#endif

        // 互斥锁
        std::unique_lock<std::mutex> lock(t1_mutex);
        // 如果缓存满了就等待
        t1_not_full.wait(lock, []{
            return t1_frame.size() < Buffer_Size;
            });
        // 增加一个元素
        t1_frame.push(frame);
        // t1_frame.push(adjust);
        // 通知下一线程开始
        t1_not_empty.notify_one();
    }
}

void ProcImg::predictFrame()
{
    cv::Mat frame;
    while (true)
    {
        // 检查是否退出
        if (this->quit)
        {
            std::cout << "线程2退出" << std::endl;
            break;
        }

        // 使用{} 限制作用域，否则锁会在一次循环结束后才释放
        {
            // 互斥锁
            std::unique_lock<std::mutex> lock(t1_mutex);
            // 如果缓存为空，就等待
            t1_not_empty.wait(lock, []{
                return !t1_frame.empty();
            });
            // 取出一个元素
            frame = t1_frame.front();
            t1_frame.pop();
            // 通知上一线程开始
            t1_not_full.notify_one();
        }

#ifdef SHOW_OUTPUT
        // 深拷贝一帧作为结果图
        cv::Mat output;
        frame.copyTo(output);
        t_show.push(output);
#endif

#ifdef TIME
        auto t1 = std::chrono::steady_clock::now(); // 计时开始
#endif

#ifdef CPU_PREPROCESS
        auto mfloat = this->infer->CPU_preprocess(frame);
#endif

#ifdef GPU_PREPROCESS
        auto mfloat = this->infer->GPU_preprocess(frame);
#endif

        void *input_data = mfloat.ptr<void>(0);
        float *output_data = this->infer->predict(input_data);

#ifdef TIME
        // 计时结束
        auto t2 = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        if (duration.count() >= 10)
            std::cout << "thread2 cost time : " << duration.count()  << " ms" << std::endl;
#endif

        // 使用{} 限制作用域，否则锁会在一次循环结束后才释放
        {
            // 互斥锁
            std::unique_lock<std::mutex> lock2(t2_mutex);
            // 如果缓存满了就等待
            t2_not_full.wait(lock2, []{
                return t2_buffers.size() < Buffer_Size;
            });
            // 增加一个元素
            t2_buffers.push(output_data);
            // 通知下一线程开始
            t2_not_empty.notify_one();
        }
    }
}

void ProcImg::getResult()
{
    float *output_data;
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
            output_data = t2_buffers.front();
            t2_buffers.pop();
            // 通知上一线程开始
            t2_not_full.notify_one();
        }

#ifdef TIME
        auto t1 = std::chrono::steady_clock::now(); // 计时开始
#endif

        this->infer->postprocess(output_data);
        auto res = this->infer->get_results();
        delete output_data;

        bool is_find = this->targ.select(res);
        std::cout << "is_find : " << is_find << std::endl;
        Serial_Data sd;
        if (is_find)
        {
            auto t = this->targ.get_target();
            std::cout << "target : " << t.class_id << std::endl;
            std::vector<float> angle = sol.getAngle(t.kpoints, t.class_id);
            float dist = sol.get_distance();
            // std::cout << "angle : " << "pitch = " << angle[0] << "yaw = " << angle[1] << std::endl;
            // std::cout << "position = " << sol.get_trans() << std::endl;
            // std::cout << "distance = " << sol.get_distance() << std::endl;

            sd.is_find = 1;
            sd.pitch.f = angle[0];
            sd.yaw.f = angle[1];
            sd.dist.f = dist;
        }
        else
        {
            sd.is_find = 0;
        }
        sp.sendData(sd);

#ifdef SHOW_FPS
            this->end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->end - this->start);
            this->FPS = "FPS : " + std::to_string(1000.0 / duration.count()).substr(0, 5);
            // this->step = 0;
            this->start = this->end;
            std::cout << FPS << std::endl;
            std::cout << "cost time = " << duration.count() << std::endl;
#endif

#ifdef TIME
        // 计时结束
        auto t2 = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        if (duration.count() >= 10)
            std::cout << "thread3 cost time : " << duration.count()  << " ms" << std::endl;
#endif

#ifdef SHOW_OUTPUT
        // 使用{} 限制作用域，否则锁会在一次循环结束后才释放
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

void ProcImg::show()
{
    cv::Mat output_img;
    std::vector<result> res;
    while (true)
    {
        // 使用{} 限制作用域，否则锁会在一次循环结束后才释放
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

        bool is_find = this->targ.select(res);
        std::cout << "is_find : " << is_find << std::endl;
        Serial_Data sd;
        if (is_find)
        {
            auto t = this->targ.get_target();
            std::cout << "target : " << t.class_id << std::endl;
            std::vector<float> angle = sol.getAngle(t.kpoints, t.class_id);
            float dist = sol.get_distance();
            // std::cout << "angle : " << "pitch = " << angle[0] << "yaw = " << angle[1] << std::endl;
            // std::cout << "position = " << sol.get_trans() << std::endl;
            // std::cout << "distance = " << sol.get_distance() << std::endl;

            sd.is_find = 1;
            sd.yaw.f = angle[0];
            sd.pitch.f = angle[1];
            sd.dist.f = dist;
        }
        else
        {
            sd.is_find = 0;
        }
        sp.sendData(sd);

        this->infer->drawplot(output_img, res);

        if (!output_img.empty())
        {
            cv::putText(output_img, this->FPS, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 200, 0), 3);
            imshow("output", output_img);
        }
        char c = cv::waitKey(1);
        if (c == 27 || c == 'q' || c == 'Q')
        {
            std::cout << "线程4退出" << std::endl;
            this->quit = true;
            break;
        }
    }
}

ProcImg::~ProcImg()
{
    delete infer;
}
