#include "inference.h"

inference::inference(const int width, const int height): 
    img_width(width), img_height(height)
{
    // load engine模型
    std::string engineFilename = "../res/armo_pose_200n_sim_fp16.engine";
    std::ifstream engineFile(engineFilename, std::ios::binary);
    assert(!engineFile.fail());
    engineFile.seekg(0, std::ios::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize); // 读取engine文件
    engineFile.close();

    this->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());   // 创建一个runtime类

    // Set the device index
    auto ret = cudaSetDevice(0);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(0) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    this->mEngine = this->runtime->deserializeCudaEngine(engineData.data(), fsize); // 反序列化engine
    assert(this->mEngine);
    engineData.clear();

    this->context = this->mEngine->createExecutionContext(); // 创建可执行上下文
    assert(this->context);

    // 创建cuda流
    assert(cudaStreamCreate(&this->ToD_stream) == cudaSuccess);
    assert(cudaStreamCreate(&this->infer_stream) == cudaSuccess);
    assert(cudaStreamCreate(&this->ToH_stream) == cudaSuccess);

    // 设置输入输出层
    for (int i = 0; i < this->mEngine->getNbIOTensors(); ++i)
    {
        this->IO_Name[i] = this->mEngine->getIOTensorName(i); // 层名称
        assert(this->mEngine->getTensorDataType(this->IO_Name[i]) == nvinfer1::DataType::kFLOAT);
        auto IO_Dims = this->context->getTensorShape(this->IO_Name[i]);
        this->IO_Size[i] = std::accumulate(IO_Dims.d, IO_Dims.d + IO_Dims.nbDims,
                                       1, std::multiplies<int64_t>()) * sizeof(float); // 计算层大小
        // Allocate CUDA memory
        this->IO_Buf[i] = nullptr;
        assert(cudaMalloc(&this->IO_Buf[i], this->IO_Size[i]) == cudaSuccess);
        this->context->setTensorAddress(this->IO_Name[i], this->IO_Buf[i]);
    }

    // 设置缩放因子
    this->r = std::min(1.0 * input_width / img_width, 1.0 * input_height / img_height);

    // 设置调色盘
    srand((unsigned)time(nullptr));
    for (int i = 0; i < nc; i++)
    {
        cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
        this->palette.emplace_back(color);
    }
}

inference::~inference()
{
    // delete this->output_data;
    cudaFree(this->IO_Buf[0]);
    cudaFree(this->IO_Buf[1]);
    // 关闭cuda流
    cudaStreamSynchronize(this->ToD_stream);
    cudaStreamSynchronize(this->infer_stream);
    cudaStreamSynchronize(this->ToH_stream);
    cudaStreamDestroy(this->ToD_stream);
    cudaStreamDestroy(this->infer_stream);
    cudaStreamDestroy(this->ToH_stream);
}

float *inference::predict(void *input_data)
{
    // 在host上开辟内存存放结果
    float *output_data = new float[this->IO_Size[1] / sizeof(float)];

    // Copy image data to device
#ifdef CPU_PREPROCESS
    assert(cudaMemcpyAsync(this->IO_Buf[0], input_data, this->IO_Size[0],
                           cudaMemcpyHostToDevice, this->ToD_stream) == cudaSuccess);
#endif

#ifdef GPU_PREPROCESS
    assert(cudaMemcpyAsync(this->IO_Buf[0], input_data, this->IO_Size[0],
                           cudaMemcpyDeviceToDevice, this->ToD_stream) == cudaSuccess);
#endif

    // inference
    assert(this->context->enqueueV3(this->infer_stream));

    // Copy predictions from device to host
    assert(cudaMemcpyAsync(output_data, this->IO_Buf[1], this->IO_Size[1],
                           cudaMemcpyDeviceToHost, this->ToH_stream) == cudaSuccess);

    return output_data;
}

cv::cuda::GpuMat inference::GPU_preprocess(cv::Mat img)
{
    // 将图片变为letter box
    cv::cuda::Stream preproc_stream;
    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(img, preproc_stream);
    int unpad_w = this->r * gpu_img.cols;
    int unpad_h = this->r * gpu_img.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(gpu_img, re, re.size());
    cv::cuda::GpuMat letterbox(input_height, input_width, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(letterbox(cv::Rect((input_width - re.cols) / 2, (input_height - re.rows) / 2, re.cols, re.rows)));

    // 转换成(b, c, h, w)(NCHW)类型
    int channel_size = input_width * input_height;
    cv::cuda::GpuMat gpu_dst(1, channel_size * input_channel, CV_8U);
    std::vector<cv::cuda::GpuMat> channels{
        cv::cuda::GpuMat(letterbox.size(), CV_8U, &gpu_dst.ptr(0)[0]),
        cv::cuda::GpuMat(letterbox.size(), CV_8U, &gpu_dst.ptr(0)[channel_size]),
        cv::cuda::GpuMat(letterbox.size(), CV_8U, &gpu_dst.ptr(0)[channel_size * 2])
    };
    cv::cuda::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
    cv::cuda::split(letterbox, channels);   // HWC -> CHW
    cv::cuda::GpuMat mfloat;
    gpu_dst.convertTo(mfloat, CV_32F, 1 / 255.f);

    return mfloat;
}

cv::Mat inference::CPU_preprocess(cv::Mat img)
{
    // 将图片变为letter box
    int unpad_w = this->r * img.cols;
    int unpad_h = this->r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat letterbox(input_height, input_width, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(letterbox(cv::Rect((input_width - re.cols) / 2, (input_height - re.rows) / 2, re.cols, re.rows)));

    // 转换成(b, c, h, w)(NCHW)类型
    int channel_size = input_width * input_height;
    cv::Mat dst(1, channel_size * input_channel, CV_8U);
    std::vector<cv::Mat> channels{
        cv::Mat(letterbox.size(), CV_8U, &dst.ptr(0)[0]),
        cv::Mat(letterbox.size(), CV_8U, &dst.ptr(0)[channel_size]),
        cv::Mat(letterbox.size(), CV_8U, &dst.ptr(0)[channel_size * 2])
    };
    cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
    split(letterbox, channels);   // HWC -> CHW
    cv::Mat mfloat;
    dst.convertTo(mfloat, CV_32F, 1 / 255.f);

    return mfloat;
}

void inference::postprocess(float *output_data)
{
    // 获得所有输出的结果
    std::vector<armo_classes> classes_ids;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<std::vector<cv::Point>> kpoints;
    for (int j = 0; (j < cols); j++)
    {
        float max_conf = 0.0f;
        int class_id;
        for (int i = 4; i < 18; i++)
        {
            float conf = *(output_data + cols * i + j);
            if (max_conf < conf)
            {
                max_conf = conf;
                class_id = i - 4;
            }
        }
        if (max_conf >= conf_thred)
        {
            classes_ids.emplace_back((armo_classes)class_id); // 类别
            scores.emplace_back(max_conf);                    // 置信度

            cv::Rect bbox; // 物体检测框
            float x0 = (input_width - r * this->img_width) / 2.0;
            float y0 = (input_height -r * this->img_height) / 2.0;
            float x = *(output_data + j) - x0;
            float y = *(output_data + j + cols) - y0;
            float w = *(output_data + j + cols * 2);
            float h = *(output_data + j + cols * 3);
            bbox.x = int((x - w / 2) / this->r);
            bbox.y = int((y - h / 2) / this->r);
            bbox.width = int(w / this->r);
            bbox.height = int(h / this->r);
            bboxes.emplace_back(bbox);

            std::vector<cv::Point> kpoint; // 关键点
            cv::Point lt;
            lt.x = int((*(output_data + j + cols * 18) - x0) / this->r);
            lt.y = int((*(output_data + j + cols * 19) - y0) / this->r);
            kpoint.push_back(lt);
            cv::Point lb;
            lb.x = int((*(output_data + j + cols * 20) - x0) / this->r);
            lb.y = int((*(output_data + j + cols * 21) - y0) / this->r);
            kpoint.push_back(lb);
            cv::Point rb;
            rb.x = int((*(output_data + j + cols * 22) - x0) / this->r);
            rb.y = int((*(output_data + j + cols * 23) - y0) / this->r);
            kpoint.push_back(rb);
            cv::Point rt;
            rt.x = int((*(output_data + j + cols * 24) - x0) / this->r);
            rt.y = int((*(output_data + j + cols * 25) - y0) / this->r);
            kpoint.push_back(rt);
            kpoints.push_back(kpoint);
        }
    }
    // 筛选检测结果
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, conf_thred, nms_thred, indices);
    // 将结果装在新容器中
    this->results.clear();    // 清空之前的结果
    while (!indices.empty())
    {
        int idx = *(--indices.end());
        indices.pop_back();
        result res;
        res.class_id = classes_ids[idx];
        res.conf = scores[idx];
        res.bbox = bboxes[idx];
        res.kpoint = kpoints[idx];
        this->results.emplace_back(res);
    }
}

void inference::drawplot(cv::Mat &output, std::vector<result> res)
{

    int size = res.size();
    for (int i = 0; i < size; i++)
    {
        std::string armo; // 类别
        switch (res[i].class_id)
        {
        case B1:
            armo = "B1";
            break;
        case B2:
            armo = "B2";
            break;
        case B3:
            armo = "B3";
            break;
        case B4:
            armo = "B4";
            break;
        case B5:
            armo = "B5";
            break;
        case BO:
            armo = "BO";
            break;
        case BS:
            armo = "BS";
            break;
        case R1:
            armo = "R1";
            break;
        case R2:
            armo = "R2";
            break;
        case R3:
            armo = "R3";
            break;
        case R4:
            armo = "R4";
            break;
        case R5:
            armo = "R5";
            break;
        case RO:
            armo = "RO";
            break;
        case RS:
            armo = "RS";
            break;
        }
        float conf = res[i].conf;                      // 置信度
        cv::Rect bbox = res[i].bbox;                   // 检测框
        std::vector<cv::Point> kpoint = res[i].kpoint; // 关键点
        // 绘制结果
        cv::Scalar color = palette[res[i].class_id];
        cv::rectangle(output, bbox, color, 2); // 检测框
        for (int j = 0; j < nk; j++)               // 关键点
        {
            cv::circle(output, res[i].kpoint[j], 3, color, -1);
        }
        std::string label = armo + " : " + std::to_string(conf).substr(0, 4);
        int *baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, baseline);
        int label_x = bbox.x;
        int label_y = bbox.y - 10 > label_size.height ? bbox.y - 10 : bbox.y + 10;
        cv::rectangle(output, cv::Point(label_x, label_y - label_size.height),
                      cv::Point(label_x + label_size.width, label_y + label_size.height),
                      color, cv::FILLED);
        cv::putText(output, label, cv::Point(label_x, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv::LINE_AA);
    }
}

std::vector<result> inference::get_results()
{
    return this->results;
}
