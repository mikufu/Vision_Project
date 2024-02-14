#include "inference.h"

inference::inference(const int img_width, const int img_height)
{
    // load engine模型
    std::string engineFilename = "../res/armo_pose_200n_sim.engine";
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
    assert(cudaStreamCreate(&this->infer_stream) == cudaSuccess);

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
        // this->context->setTensorAddress(this->IO_Name[i], this->IO_Buf[i]);
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

    // 为host_output开辟空间，用于存放结果
    this->output_data = new float[this->IO_Size[1] / sizeof(float)];

    // Run TensorRT inference
    // assert(this->context->enqueueV3(this->infer_stream));
}

inference::~inference()
{
    delete this->output_data;
    cudaFree(this->IO_Buf[0]);
    cudaFree(this->IO_Buf[1]);
    // delete this->mEngine;
    // delete this->runtime;
    // delete this->context;
    cudaStreamSynchronize(this->infer_stream);
    cudaStreamDestroy(this->infer_stream);
}

void inference::predict(cv::Mat img)
{
    img.copyTo(this->img_result);
    auto mfloat = this->preprocess(img);
    auto input_data = mfloat.ptr<void>();

    // Copy image data to device
    assert(cudaMemcpyAsync(this->IO_Buf[0], input_data, this->IO_Size[0],
                           cudaMemcpyDeviceToDevice, this->infer_stream) == cudaSuccess);

    // inference
    assert(this->context->executeV2(IO_Buf));

    // Copy predictions from output binding memory
    assert(cudaMemcpyAsync(this->output_data, this->IO_Buf[1], this->IO_Size[1],
                           cudaMemcpyDeviceToHost, this->infer_stream) == cudaSuccess);

    postprocess();
    // this->drawplot();
}

cv::cuda::GpuMat inference::preprocess(cv::Mat img)
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

void inference::postprocess()
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
            float conf = *(this->output_data + cols * i + j);
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
            float x0 = (input_width - r * this->img_result.cols) / 2.0;
            float y0 = (input_height -r * this->img_result.rows) / 2.0;
            float x = *(this->output_data + j) - x0;
            float y = *(this->output_data + j + cols) - y0;
            float w = *(this->output_data + j + cols * 2);
            float h = *(this->output_data + j + cols * 3);
            bbox.x = int((x - w / 2) / this->r);
            bbox.y = int((y - h / 2) / this->r);
            bbox.width = int(w / this->r);
            bbox.height = int(h / this->r);
            bboxes.emplace_back(bbox);

            std::vector<cv::Point> kpoint; // 关键点
            cv::Point lt;
            lt.x = int((*(this->output_data + j + cols * 18) - x0) / this->r);
            lt.y = int((*(this->output_data + j + cols * 19) - y0) / this->r);
            kpoint.push_back(lt);
            cv::Point lb;
            lb.x = int((*(this->output_data + j + cols * 20) - x0) / this->r);
            lb.y = int((*(this->output_data + j + cols * 21) - y0) / this->r);
            kpoint.push_back(lb);
            cv::Point rb;
            rb.x = int((*(this->output_data + j + cols * 22) - x0) / this->r);
            rb.y = int((*(this->output_data + j + cols * 23) - y0) / this->r);
            kpoint.push_back(rb);
            cv::Point rt;
            rt.x = int((*(this->output_data + j + cols * 24) - x0) / this->r);
            rt.y = int((*(this->output_data + j + cols * 25) - y0) / this->r);
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

void inference::drawplot()
{
    int size = this->results.size();
    for (int i = 0; i < size; i++)
    {
        std::string armo; // 类别
        switch (results[i].class_id)
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
        float conf = results[i].conf;                      // 置信度
        cv::Rect bbox = results[i].bbox;                   // 检测框
        std::vector<cv::Point> kpoint = results[i].kpoint; // 关键点
        // 绘制结果
        cv::Scalar color = palette[results[i].class_id];
        cv::rectangle(this->img_result, bbox, color, 2); // 检测框
        for (int j = 0; j < nk; j++)               // 关键点
        {
            cv::circle(this->img_result, results[i].kpoint[j], 3, color, -1);
        }
        std::string label = armo + " : " + std::to_string(conf).substr(0, 4);
        int *baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, baseline);
        int label_x = bbox.x;
        int label_y = bbox.y - 10 > label_size.height ? bbox.y - 10 : bbox.y + 10;
        cv::rectangle(this->img_result, cv::Point(label_x, label_y - label_size.height),
                      cv::Point(label_x + label_size.width, label_y + label_size.height),
                      color, cv::FILLED);
        cv::putText(this->img_result, label, cv::Point(label_x, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv::LINE_AA);
    }
}

std::vector<result> inference::get_results()
{
    return this->results;
}

cv::Mat inference::get_img_result()
{
    return this->img_result;
}
