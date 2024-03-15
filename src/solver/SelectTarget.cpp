#include "SelectTarget.hpp"

bool target::select(const std::vector<result> &results)
{
    int idx = -1;
    float max_area = 0.0;
    // 找到最近的或者倾斜最小的装甲板
    if(enermy)  // 筛选红方
    {
        for (int i = 0; i < results.size(); i++)
        {
            if (RS >= results[i].class_id
                && R1 <= results[i].class_id)
            {
                float area = this->target_area(results[i].kpoints);
                // float rate = this->wh_rate(results[i].kpoints);
                if (area > MIN_AREA /*&& rate >0.5*/ && area > max_area)
                {
                    max_area = area;
                    idx = i;
                }
            }
            
        }
    }
    else    // 筛选蓝方
    {
        for (int i = 0; i < results.size(); i++)
        {
            if (BS >= results[i].class_id
                && B1 <= results[i].class_id)
            {
                float area = this->target_area(results[i].kpoints);
                // float rate = this->wh_rate(results[i].kpoints);
                // std::cout << "wh_rate = " << rate << std::endl;
                if (area > MIN_AREA /*&& rate >0.5*/ && area > max_area)
                {
                    max_area = area;
                    idx = i;
                }
            }
            
        }
    }
    return anti_switch(results, idx);
}

float target::target_area(const std::vector<cv::Point2f> &kpoints)
{
    return cv::contourArea(kpoints);
}

float target::wh_rate(const std::vector<cv::Point2f> &kpoints)
{
    auto lt = kpoints[0];
    auto lb = kpoints[1];
    auto rb = kpoints[2];
    auto rt = kpoints[3];

    float rate1 = sqrt((lt.x - rt.x) * (lt.x - rt.x) + (lt.y - rt.y) * (lt.y - rt.y))
            / sqrt((lt.x - lb.x) * (lt.x - lb.x) + (lt.y - lb.y) * (lt.y - lb.y));
        
    return rate1;
}

bool target::anti_switch(const std::vector<result> &results, int idx)
{
    if (idx != -1)  // 找到目标
    {
        this->tg = results[idx];
        if (tg.class_id != last_tg.class_id)
        {
            if (++switch_cnt % MAX_COUNT == 0)
            {
                last_tg = tg;   // 切换目标
            }
            else if (is_find == true)
            {
                tg.class_id = last_tg.class_id;
            }
            else
            {
                last_tg = tg;
            }
        }
        is_find = true;

        return true;
    }
    else    // 未找到目标
    {
        if (is_find == true)
        {
            if (++switch_cnt % MAX_COUNT == 0)
            {
                is_find = false;
            }
            else
            {
                tg = last_tg;
                return true;
            }
        }

        return false;
    }
}
