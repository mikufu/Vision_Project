#ifndef __SERIAL_H_
#define __SERIAL_H_

#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/ioctl.h>

#define BAUD 115200 // 波特率
#define SERIAL_DEV "/dev/ttyUSB0"  // 串口设备名

typedef union
{
    float f;
    unsigned char c[4];
}float2uchar;

typedef struct
{
    unsigned char is_find;    // 是否找到目标
    float2uchar yaw;    // 目标相对云台(枪口)的yaw轴
    float2uchar pitch;  // 目标相对云台(枪口)的pitch轴
    float2uchar dist;   // 目标相对云台(枪口)的距离
}Serial_Data;

class serial_port
{
private:
    int fd; // 串口号
    int speed, databits, stopbits, parity;
    Serial_Data r_data; // 原数据
    unsigned char t_data[20];   // 发送的数据
public:
    serial_port();

    ~serial_port();

    bool init_serial();

    void sendData(const Serial_Data &data);    // 发送数据
private:
    bool set_Baud();    // 设置波特率

    bool set_Bit(); // 设置bit位

    void transformData();   // 转换数据
};

#endif