#pragma once

#include "inference.h"

#define ENERMY 1    // 0表示敌方为蓝，1表示敌方为红
#define MAX_COUNT 3 // 切换目标所等待的帧数
#define MIN_AREA 1000   

class target
{
private:
    result tg;      // 选择的目标
    result last_tg;   // 上一目标
    int switch_cnt; // 防止频繁切换目标
    bool is_find;

public:
    bool select(const std::vector<result> &results);

    const result get_target() const
    {
        return this->tg;
    }

private:
    float target_area(const std::vector<cv::Point2f> &kpoints);

    float wh_rate(const std::vector<cv::Point2f> &kpoints);

    bool anti_switch(const std::vector<result> &results, int idx); // 防止频繁切换目标
};
