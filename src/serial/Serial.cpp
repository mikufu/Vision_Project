#include "Serial.h"

serial_port::serial_port()
{
    fd = open(SERIAL_DEV, O_RDWR | O_NOCTTY | O_NDELAY);  // 打开设备

    speed = BAUD;   // 波特率
    databits = 8;   // 数据位
    stopbits = 1;   // 停止位
    parity = 's';   // 奇偶校验位
}

serial_port::~serial_port()
{
    close(fd);
}

bool serial_port::init_serial()
{
    if (fd == -1)
    {
        perror(SERIAL_DEV);
        return false;
    }

    if (set_Baud() == false)
        return false;

    if (set_Bit() == false)
        return false;

    printf("open serial successed\n");

    return true;
}

void serial_port::sendData(const Serial_Data &data)
{
    this->r_data = data;

    transformData();

    size_t bytes;
    if (r_data.is_find == 0)
        bytes = 3;
    else
        bytes = 15;

    write(fd, t_data, bytes);
}

bool serial_port::set_Baud()
{
    // 设置波特率
    int speed_arr[] = {B115200, B38400, B19200, B9600, B4800, B2400, B1200, B300};
    int name_arr[] = {115200, 38400, 19200, 9600, 4800, 2400, 1200, 300};
    
    int i;
    int status;
    termios opt;
    tcgetattr(fd, &opt);
    for (i = 0; i < sizeof(speed_arr) / sizeof(int); i++)
    {
        if (speed == name_arr[i])
        {
            tcflush(fd, TCIOFLUSH); // 清空缓冲区内容
            cfsetispeed(&opt, speed_arr[i]);
            cfsetospeed(&opt, speed_arr[i]);    // 设置接收和发送的波特率
            status = tcsetattr(fd, TCSANOW, &opt);  // 使设置立即生效
            if (status != 0)
            {
                perror("tcsetattr fd1");
                return false;
            }
            tcflush(fd, TCIOFLUSH);
        }
    }
    return true;
}

bool serial_port::set_Bit()
{
     // 设置比特位
    termios termios_p;
    if (tcgetattr(fd, &termios_p) != 0)
    {
        perror("SetupSerial 1");
        return false;
    }

    termios_p.c_cflag |= (CLOCAL | CREAD);  // 接受数据
    termios_p.c_cflag &= ~CSIZE;    // 设置数据位数

    switch (databits)
    {
    case 7:
        termios_p.c_cflag |= CS7;
        break;
    case 8:
        termios_p.c_cflag |= CS8;
        break;
    default:
        fprintf(stderr, "Unsupported data size\n");
        return false;
    }

    // 设置奇偶校验位
    switch (parity)
    {
    case 'n':
    case 'N':
        termios_p.c_cflag &= ~PARENB;   /* Clear parity enable */
        termios_p.c_iflag &= ~INPCK;    /* Enable parity checking */
        break;
    case 'o':
    case 'O':
        termios_p.c_cflag |= (PARENB | PARODD); /* 设置为奇校验位 */
        termios_p.c_iflag |= INPCK; /* Disable parity checking */
        break;
    case 'e':
    case 'E':
        termios_p.c_cflag |= PARENB;    /* Enable parity */
        termios_p.c_cflag &= ~PARODD;   /* 转换为偶校验 */
        termios_p.c_iflag |= INPCK; /* Disable parity checking */
        break;
    case 's':
    case 'S':   /* as no parity */
        termios_p.c_cflag &= ~PARENB;
        termios_p.c_cflag &= ~CSTOPB;
        break;
    default:
        fprintf(stderr, "Unsupported parity\n");
        return false;
    }

    // 设置停止位
    switch (stopbits)
    {
    case 1:
        termios_p.c_cflag &= ~CSTOPB;
        break;
    case 2:
        termios_p.c_cflag |= CSTOPB;
        break;
    default:
        fprintf(stderr, "Unsupported stop bits\n");
        return false;
    }

    /* Set input parity option */
    if (parity != 'n')
        termios_p.c_iflag |= INPCK;
    
    tcflush(fd, TCIFLUSH);  // 清除输入缓冲区
    termios_p.c_cc[VTIME] = 150;    // 设置超时 15s
    termios_p.c_cc[VMIN] = 0;   // 设置最小接受字符
    termios_p.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);   // input原始输入
    termios_p.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    termios_p.c_iflag &= ~(ICRNL | IGNCR);
    termios_p.c_oflag &= ~OPOST;    /* output禁用输出处理*/

    if (tcsetattr(fd, TCSANOW, &termios_p) != 0)    // update the options and do it now
    {
        perror("Setup Serial 3");
        return false;
    }
    
    return true;
}

void serial_port::transformData()
{
    t_data[0] = 0xA5;   // 数据开始位
    t_data[1] = r_data.is_find;   // 是否找到目标
    if (t_data[1] == 0)
        t_data[2] = 0xFF;   // 数据结束位
    else
    {
        // yaw轴数据
        t_data[2] = r_data.yaw.c[0];
        t_data[3] = r_data.yaw.c[1];
        t_data[4] = r_data.yaw.c[2];
        t_data[5] = r_data.yaw.c[3];
        // pitch轴数据
        t_data[6] = r_data.pitch.c[0];
        t_data[7] = r_data.pitch.c[1];
        t_data[8] = r_data.pitch.c[2];
        t_data[9] = r_data.pitch.c[3];
        // 距离数据
        t_data[10] = r_data.dist.c[0];
        t_data[11] = r_data.dist.c[1];
        t_data[12] = r_data.dist.c[2];
        t_data[13] = r_data.dist.c[3];

        t_data[14] = 0xFF;
    }

    if (t_data[1] == 0)
    {
        for (int i = 0; i < 3; i++)
        {
            printf("%x ", t_data[i]);
        }
    }
    else
    {
        for (int i = 0; i < 15; i++)
        {
            printf("%x ", t_data[i]);
        }
    }
    printf("\n");
}
