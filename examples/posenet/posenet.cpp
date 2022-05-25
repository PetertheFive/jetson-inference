/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
/*
 * edited by:   Peter Wu
 * date:        2022.2.10
 * 
 * 
*/


#include "videoSource.h"
#include "videoOutput.h"

#include "poseNet.h"

#include <signal.h>
#include <math.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>
#include <net/if.h>
#include <stddef.h>
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
extern "C"
{
#include "tty.h"
}
#include <fcntl.h>

#include <unistd.h>
#include <pthread.h>
#include <termios.h>

#include <sys/timerfd.h>
#include <sys/time.h>
#include <time.h>

#define FWMESSAGE "\n FW: 0.0.0.20220210a\n\n"

#define BODYEXISTTIME 10    // How many time when there is no body detection should the program think there is no human. Unit: frame, such as 40ms or 80ms.

/* animation fade-in degree control 
*   Logic: when frame number is bigger than threshold 1, the variation of each frame will be degree 1. Only the biggest threshold will take effect
*/
#define DEGREE1 	30
#define THRESHOLD1 	50

#define DEGREE2 	10
#define THRESHOLD2 	15

#define DEGREE3 	3
#define THRESHOLD3 	5

#define DEGREE4 	1
#define THRESHOLD4 	0   //last threshold should be 0
/* */

#define BAUDRATE B115200
#define DEV_NAME "/dev/ttyTHS1"

#define handle_error(msg)   \
    do                      \
    {                       \
        perror(msg);        \
        exit(EXIT_FAILURE); \
    } while (0)

/* function declaration */
void sendFrameNum(int frameNum);
int frameSmoothControll(int now, int target);
void playwithnobody(void);
int position_legal_detection(int coordinate_X, int coordinate_Y);

/* variate declaration */

int bodyPoint = 011516;
/*
    abcdef:
    ab = method; cd = point 1; ef = point 2 ...

    example:
    011516: use left and right ankles with average y coordinate

    method:
        1. average y coordinate of points
        2. distance between points

    body point:

	int nose = pose->FindKeypoint(0);
	int leftEye = pose->FindKeypoint(1);
	int rightEye = pose->FindKeypoint(2);
	int leftEar = pose->FindKeypoint(3);
	int rightEar = pose->FindKeypoint(3);
    int leftShoulder = pose->FindKeypoint(5);
    int rightShoulder = pose->FindKeypoint(6);
    int leftElbow = pose->FindKeypoint(7);
    int leftWrist = pose->FindKeypoint(9);
    int rightElbow = pose->FindKeypoint(8);
    int rightWrist = pose->FindKeypoint(10);
    int leftHip = pose->FindKeypoint(11);
    int rightHip = pose->FindKeypoint(12);
    int leftKnee = pose->FindKeypoint(13);
    int leftAnkle = pose->FindKeypoint(15);
    int rightKnee = pose->FindKeypoint(14);
    int rightAnkle = pose->FindKeypoint(16);
*/

int nobodyCounter = BODYEXISTTIME;

int posDegree = 0; 
int posDegree_r = 0;
int frameNum = 0;
int frameTarget = 0;

/* body positon limitation,  unit: pixel 
    default for all position in camera
*/
int pos_x_Min = 0;       //-720
int pos_x_Max = 1280;       //1280
int disArg_x = 0;
int pos_y_Min = 0;        //0
int pos_y_Max = 720;        //720
int disArg_y = 0;

int bodyTarget = 0;

int posArg = 3;
int frameArg = 1;
int frame_Max = 720;         //最大帧序号
int speed_ms = 40;

static pthread_mutex_t g_mutex_lock;

int tty_fd = -1;

struct distance
{
    int value;
    unsigned long time; // 赋值时刻
};

static distance distance_num; //    距离传输值

/******************************************************************************
 system function
*******************************************************************************/

unsigned long mstime()
{
    struct timeval tv;
    unsigned long ret;
    gettimeofday(&tv, NULL);
    ret = tv.tv_usec / 1000 + tv.tv_sec * 1000;
    //printf("msTime:%ld\n", ret);
    return ret;
    //printf("printTime: current time:%ld.%ld ", tv.tv_sec, tv.tv_usec);
}

void getdistance(int *value, unsigned long *time)
{
    pthread_mutex_lock(&g_mutex_lock);
    *value = distance_num.value;
    *time = distance_num.time;
    pthread_mutex_unlock(&g_mutex_lock);
}

void setdistance(int value)
{
    pthread_mutex_lock(&g_mutex_lock);
    distance_num.value = value;
    distance_num.time = mstime();
    pthread_mutex_unlock(&g_mutex_lock);
}

//float ShoulderWidth = 0;
void *myThreadFun(void *arg);

bool signal_recieved = false;

void sig_handler(int signo)
{
    if (signo == SIGINT)
    {
        LogVerbose("received SIGINT\n");
        signal_recieved = true;
    }
}

/******************************************************************************
  Function:       tty_init
  Description:    initialize tty device
  Input:          tty    	--  tty device name, such as '/dev/ttyO0', '/dev/ttyO5'

  Output:
  Return:         int		-- return the tty fd
  Others:         NONE
*******************************************************************************/
int tty_init(const char *tty)
{
    int fd;
    fd = open(tty, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0)
    {
        printf("open tty");
    }
    return fd;
}

/******************************************************************************
  Function:       tty_setting
  Description:    set the tty device's mode and bitrate
  Input:          fd       --  tty device fd
                  bitrate --  tty baudrate
                  mode	   --  tty mode, 1: rs485; 0: rs232
                  flow	   --  cts/rts flow control
                  par	   --  odd/even
                  stop	   --  number of stop bits
                  
  Output:
  Return:         int 	   --  tty setting status 0:success
  Others:         NONE
*******************************************************************************/
int tty_setting(int fd, int bitrate, int datasize, int mode, int flow, int par, int stop)
{
    struct termios newtio;

    /* ignore modem control lines and enable receiver */
    memset(&newtio, 0, sizeof(newtio));
    newtio.c_cflag = newtio.c_cflag |= CLOCAL | CREAD;
    if (flow == 1)
    {
        newtio.c_cflag = newtio.c_cflag |= CLOCAL | CREAD | CRTSCTS;
    }

    newtio.c_cflag &= ~CSIZE;

    /* set character size */
    switch (datasize)
    {
    case 8:
        newtio.c_cflag |= CS8;
        break;
    case 7:
        newtio.c_cflag |= CS7;
        break;
    case 6:
        newtio.c_cflag |= CS6;
        break;
    case 5:
        newtio.c_cflag |= CS5;
        break;
    default:
        newtio.c_cflag |= CS8;
        break;
    }

    /* set the parity */
    switch (par)
    {
    case 'o':
    case 'O':
        newtio.c_cflag |= PARENB;
        newtio.c_cflag |= PARODD;
        newtio.c_iflag |= (INPCK | ISTRIP);
        break;
    case 'e':
    case 'E':
        newtio.c_cflag |= PARENB;
        newtio.c_cflag &= ~PARODD;
        newtio.c_iflag |= (INPCK | ISTRIP);
        break;
    case 'n':
    case 'N':
        newtio.c_cflag &= ~PARENB;
        break;
    default:
        newtio.c_cflag &= ~PARENB;
        break;
    }

    /* set the stop bits */
    switch (stop)
    {
    case 1:
        newtio.c_cflag &= ~CSTOPB;
        break;
    case 2:
        newtio.c_cflag |= CSTOPB;
        break;
    default:
        newtio.c_cflag &= ~CSTOPB;
        break;
    }

    /* set output and input baud rate */
    switch (bitrate)
    {
    case 0:
        cfsetospeed(&newtio, B0);
        cfsetispeed(&newtio, B0);
        break;
    case 50:
        cfsetospeed(&newtio, B50);
        cfsetispeed(&newtio, B50);
        break;
    case 75:
        cfsetospeed(&newtio, B75);
        cfsetispeed(&newtio, B75);
        break;
    case 110:
        cfsetospeed(&newtio, B110);
        cfsetispeed(&newtio, B110);
        break;
    case 134:
        cfsetospeed(&newtio, B134);
        cfsetispeed(&newtio, B134);
        break;
    case 150:
        cfsetospeed(&newtio, B150);
        cfsetispeed(&newtio, B150);
        break;
    case 200:
        cfsetospeed(&newtio, B200);
        cfsetispeed(&newtio, B200);
        break;
    case 300:
        cfsetospeed(&newtio, B300);
        cfsetispeed(&newtio, B300);
        break;
    case 600:
        cfsetospeed(&newtio, B600);
        cfsetispeed(&newtio, B600);
        break;
    case 1200:
        cfsetospeed(&newtio, B1200);
        cfsetispeed(&newtio, B1200);
        break;
    case 1800:
        cfsetospeed(&newtio, B1800);
        cfsetispeed(&newtio, B1800);
        break;
    case 2400:
        cfsetospeed(&newtio, B2400);
        cfsetispeed(&newtio, B2400);
        break;
    case 4800:
        cfsetospeed(&newtio, B4800);
        cfsetispeed(&newtio, B4800);
        break;
    case 9600:
        cfsetospeed(&newtio, B9600);
        cfsetispeed(&newtio, B9600);
        break;
    case 19200:
        cfsetospeed(&newtio, B19200);
        cfsetispeed(&newtio, B19200);
        break;
    case 38400:
        cfsetospeed(&newtio, B38400);
        cfsetispeed(&newtio, B38400);
        break;
    case 57600:
        cfsetospeed(&newtio, B57600);
        cfsetispeed(&newtio, B57600);
        break;
    case 115200:
        cfsetospeed(&newtio, B115200);
        cfsetispeed(&newtio, B115200);
        break;
    case 230400:
        cfsetospeed(&newtio, B230400);
        cfsetispeed(&newtio, B230400);
        break;
    default:
        cfsetospeed(&newtio, B9600);
        cfsetispeed(&newtio, B9600);
        break;
    }

    /* set timeout in deciseconds for non-canonical read */
    newtio.c_cc[VTIME] = 0;
    /* set minimum number of characters for non-canonical read */
    newtio.c_cc[VMIN] = 1;

    /* flushes data received but not read */
    tcflush(fd, TCIFLUSH);
    /* set the parameters associated with the terminal from
       the termios structure and the change occurs immediately */
    if ((tcsetattr(fd, TCSANOW, &newtio)) != 0)
    {
        printf("set_tty/tcsetattr");
        return -1;
    }

    //	tty_mode(fd, mode);
    return 0;
}

static int get_device_status(int fd)
{
    struct termios t;
    if (fd < 0)
        return 0;
    //  if (portfd_is_socket && portfd_is_connected)
    //    return 1;
    return !tcgetattr(fd, &t);
}

int tty_read(int fd, char *frame)
{
    struct timeval tv;
    fd_set fds;
    int ret;

    unsigned int timeout = TTY_READ_TIMEOUT_USEC; //
    tv.tv_sec = 0;
    tv.tv_usec = timeout;

    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    if (timeout > 0)
    {
        ret = select(fd + 1, &fds, NULL, NULL, &tv);
        if (ret < 0)
        {
            printf("select tty");
        }
        else if (ret == 0)
        {
            //			 select timeout
            //			 dbg_printf("select %d timeout\r\n", fd);
            //               printf("select tty");
        }
    }
    //    if (!get_device_status(portfd_connected)) {//minicom
    //        /* Ok, it's gone, most probably someone unplugged the USB-serial, we
    //         * need to free the FD so that a replug can get the same device
    //         * filename, open it again and be back */
    //        int reopen = portfd == -1;
    //        close(portfd);
    //        lockfile_remove();
    //        portfd = -1;
    //        if (open_term(reopen, reopen, 1) < 0) {
    //          if (!error_on_open_window)
    //            error_on_open_window = mc_tell(_("Cannot open %s!"), dial_tty);
    //        } else {
    //          if (error_on_open_window) {
    //            mc_wclose(error_on_open_window, 1);
    //            error_on_open_window = NULL;
    //          }
    //        }
    //      }

    if (FD_ISSET(fd, &fds))
    {
        ret = read(fd, frame, TTY_READ_BUFFER_SIZE);
        if (ret < 0)
        {
            perror("tty read:");
            //  printf("tty break off\r\n");
        }
    }

    return ret;
}

int tty_write(int fd, char *frame, int len)
{
    int ret = -1;
    ret = write(fd, frame, len);
    if (ret < 0)
    {
        //printf("tty write!\n");
    }
    return ret;
}

int tty_mode(const int fd, int mode)
{
    struct serial_rs485 rs485conf;
    int res;
    /* Get configure from device */
    res = ioctl(fd, TIOCGRS485, &rs485conf);
    if (res < 0)
    {
        perror("Ioctl error on getting 485 configure:");
        close(fd);
        return res;
    }

    /* Set enable/disable to configure */
    if (mode == TTY_RS485_MODE)
    {
        // Enable rs485 mode
        rs485conf.flags |= SER_RS485_ENABLED;
    }
    else
    {
        // Disable rs485 mode
        rs485conf.flags &= ~(SER_RS485_ENABLED);
    }

    rs485conf.delay_rts_before_send = 0x00000004;

    /* Set configure to device */
    res = ioctl(fd, TIOCSRS485, &rs485conf);
    if (res < 0)
    {
        perror("Ioctl error on setting 485 configure:");
        close(fd);
    }

    return res;
}

void static setusart(int tty_fd)
{
    fcntl(tty_fd, F_SETFL, 0); //重设为堵塞状态, 去掉O_NONBLOCK
    struct termios opts;
    tcgetattr(tty_fd, &opts); //把原设置获取出来，存放在opts

    cfsetispeed(&opts, BAUDRATE);
    cfsetospeed(&opts, BAUDRATE);

    opts.c_cflag |= CLOCAL | CREAD; //忽略modem控制线, 启动接收器

    // 8N1
    opts.c_cflag &= ~PARENB;
    opts.c_cflag &= ~CSTOPB;
    opts.c_cflag |= CS8;

    opts.c_cflag &= ~CRTSCTS; //关闭硬件流控
    opts.c_iflag &= ~(ICRNL | IXON);

    opts.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); //raw input
    opts.c_oflag &= ~OPOST;                          // raw output
    opts.c_cc[VTIME] = 0;                            /* Setting Time outs */
    opts.c_cc[VMIN] = 1;
    tcsetattr(tty_fd, TCSANOW, &opts);
    printf("Initialize tty device %s success!\r\n", DEV_NAME);
}

int usage()
{
    printf("usage: posenet [--help] [--network=NETWORK] ...\n");

    printf("                input_URI [output_URI]\n\n");
    printf("Run pose estimation DNN on a video/image stream.\n");
    printf("See below for additional arguments that may not be shown above.\n\n");
    printf("positional arguments:\n");
    printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
    printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

    printf("%s", poseNet::Usage());
    printf("%s", videoSource::Usage());
    printf("%s", videoOutput::Usage());
    printf("%s", Log::Usage());

    return 0;
}

/******************************************************************************
 main function
*******************************************************************************/

int main(int argc, char **argv)
{
    printf(FWMESSAGE);

    /*
	 * parse command line
	 */
    commandLine cmdLine(argc, argv);

    if (cmdLine.GetFlag("help"))
        return usage();

    /*
	 * attach signal handler
	 */
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        LogError("can't catch SIGINT\n");

    /*
	 * create input stream
	 */
    videoSource *input = videoSource::Create(cmdLine, ARG_POSITION(0));

    if (!input)
    {
        LogError("posenet: failed to create input stream\n");
        return 0;
    }

    /*
	 * create output stream
	 */
    videoOutput *output = videoOutput::Create(cmdLine, ARG_POSITION(1));

    if (!output)
        LogError("posenet: failed to create output stream\n");

    /* switch the argc of user */
    switch(argc)
    {
        case 12:
            disArg_x = atoi(argv[11]);    
        case 11:
            pos_x_Max = atoi(argv[10]);
        case 10:
            pos_x_Min = atoi(argv[9]);
        case 9:
            disArg_y = atoi(argv[8]);
        case 8:
            pos_y_Max = atoi(argv[7]);
        case 7:
            pos_y_Min = atoi(argv[6]);
        case 6: 
            frameArg = atoi(argv[5]);
        case 5:
            posArg = atoi(argv[4]);
        case 4:
            bodyPoint = atoi(argv[3]);
        case 3:
            speed_ms = atoi(argv[2]);   //argv[0] = posenet; argv[1] /dev/video?
            break;
        default:
            LogError("posenet: no user argc input\n");
            break;
    }
    /*
	 * create recognition network
	 */
    poseNet *net = poseNet::Create(cmdLine);

    if (!net)
    {
        LogError("posenet: failed to initialize poseNet\n");
        return 0;
    }

    // parse overlay flags
    const uint32_t overlayFlags = poseNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "links,keypoints"));

    tty_fd = tty_init(DEV_NAME); //open tty

    if (tty_fd < 0)
    {
        printf("Initialize tty device %s failed!\r\n", DEV_NAME);
    }

    /*
	 * processing loop
	 */
    pthread_t pthread = 0; //创建一个子线程
    int ret = pthread_create(&pthread, NULL, myThreadFun, NULL);


    printf("\ninitial success\n\n");

    while (!signal_recieved)
    {
        // capture next image image
        uchar3 *image = NULL;
        int bodyDistance_closest = 0;

        if (!input->Capture(&image, 1000))
        {
            // check for EOS
            if (!input->IsStreaming())
                break;

            LogError("posenet: failed to capture next frame\n");
            continue;
        }

        // run pose estimation
        std::vector<poseNet::ObjectPose> poses;

        if (!net->Process(image, input->GetWidth(), input->GetHeight(), poses, overlayFlags))
        {
            LogError("posenet: failed to process frame\n");
            continue;
        }

        //LogInfo("posenet: detected %zu %s(s)\n", poses.size(), net->GetCategory());// 类别

        // render outputs
        if (output != NULL)
        {
            output->Render(image, input->GetWidth(), input->GetHeight()); //显示渲染

            // update status bar
            char str[256];
            //sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
            sprintf(str, "Full screen:  (x, y) = (0~1280, 720~0).  Limitations:  (x, y) = (%d~%d with slope %d, %d~%d with slope %d).", pos_x_Min, pos_x_Max, disArg_x, pos_y_Max, pos_y_Min, disArg_y);
            output->SetStatus(str);

            // check if the user quit
            if (!output->IsStreaming())
                signal_recieved = true;
        }


        for (std::vector<poseNet::ObjectPose>::iterator pose = poses.begin(); pose != poses.end(); ++pose)
        {
            int leftAnkle = pose->FindKeypoint(15);
            int rightAnkle = pose->FindKeypoint(16);

            if ((leftAnkle != -1) && (rightAnkle != -1))
            {
                int anklePosition_x = (int)(pose->Keypoints[leftAnkle].x + pose->Keypoints[rightAnkle].x)/2;
                int anklePosition_y = (int)(pose->Keypoints[leftAnkle].y + pose->Keypoints[rightAnkle].y)/2;

                //LogVerbose("User #%d: leftAnkle = (%f,%f) righAnkle = (%f,%f)\n", userNum, pose->Keypoints[leftAnkle].x, pose->Keypoints[leftAnkle].y, pose->Keypoints[rightAnkle].x, pose->Keypoints[rightAnkle].y);

                /* judge if the human body is in the legal area */
                if(position_legal_detection(anklePosition_x, anklePosition_y))
                {
                    bodyTarget = anklePosition_y;
                    setdistance((int)bodyTarget);
                    break;
                }
            }

        }

        //print out timing info 打印cpu占用信息
        //net->PrintProfilerTimes();
    }

    /*
	 * destroy resources
	 */
    LogVerbose("posenet: shutting down...\n");

    SAFE_DELETE(input);
    SAFE_DELETE(output);
    SAFE_DELETE(net);

    LogVerbose("posenet: shutdown complete.\n");
    return 0;
}

/*
*   LED control
*   
*/
void *myThreadFun(void *arg)
{
    struct timespec now;

    if (clock_gettime(CLOCK_REALTIME, &now) == -1)
        handle_error("clock_gettime");

    struct itimerspec new_value;
    new_value.it_value.tv_sec = now.tv_sec + 2;
    new_value.it_value.tv_nsec = now.tv_nsec;
    new_value.it_interval.tv_sec = 0;
    new_value.it_interval.tv_nsec = speed_ms * 1000000;

    int fd = timerfd_create(CLOCK_REALTIME, 0);

    if (fd == -1)
        handle_error("timerfd_create");

    if (timerfd_settime(fd, TFD_TIMER_ABSTIME, &new_value, NULL) == -1)
        handle_error("timerfd_settime");

    unsigned long last_time = 0;
    unsigned long time = 0;
    uint64_t exp;

    //printf("timer started\n");

    while (1)
    {
        ssize_t s = read(fd, &exp, sizeof(uint64_t));

        if (s != sizeof(uint64_t))
            handle_error("read");

        getdistance(&posDegree, &time); //since the top is 0 and bottom is 720.

        /* update the animation */
        if (time != last_time) //获取到了新的距离
        {
            //LogVerbose("posDegree = %d\n", posDegree);
            nobodyCounter = BODYEXISTTIME;

            /* judge if the change of pos is bigger than threshold */
            if (abs(posDegree - posDegree_r) >= posArg)
            {
                posDegree_r = posDegree;	
                frameTarget = ((posDegree-pos_y_Min)*frame_Max)/(pos_y_Max-pos_y_Min);
            }

            frameNum = frameSmoothControll(frameNum, frameTarget);
            sendFrameNum(frameNum);
            //LogVerbose("frameNum_nb = %d, frameTarget = %d\n", frameNum, frameTarget);
        }
        else
        {
	        playwithnobody();
        }

        last_time = time;
        //printf("read: %llu",exp);	
    }
}


/******************************************************************************
 user function
*******************************************************************************/

int position_legal_detection(int coordinate_X, int coordinate_Y)
{
    if(coordinate_Y > (coordinate_X * disArg_y / 100 + pos_y_Max))
        return 0;
    else if(coordinate_Y < (coordinate_X * disArg_y / 100 + pos_y_Min))
        return 0;
    else if(coordinate_X > (coordinate_Y * disArg_x / 100 + pos_x_Max))
        return 0;
    else if(coordinate_X < (coordinate_Y * disArg_x / 100 + pos_x_Min))
        return 0;
    else
        return 1;
}

void playwithnobody(void)
{
    if(nobodyCounter <= 0)
    {
        frameNum = frameNum + frameArg;
        sendFrameNum(frameNum);
        frameNum %= frame_Max;
    }
    else    
    {
        nobodyCounter--;
    }
}

void insert(unsigned char value, unsigned char *pos, int len) //插值 到第一个位置
{
    for (int j = len; j > 0; j--) //插入 从最后一个开始
    {
        pos[j] = pos[j - 1];
    }
    pos[0] = value;
}

void sendFrameNum(int frameNum)
{
    //LogVerbose("Frame number: %d\n", frameNum);
    unsigned char data[100] =
    { 0XF7,
      0X00,
      0X00,
      0X00,
      0X00,
      0X00,
      0X00,
      0X00,
      0X00,
      0X00,
      0X00,
      0X00,
      0X00,
      0X55,
      0XAA, //帧头 0~14
      0X01,
      0X01, //类型 15 16
      0X00,
      0X04, //长度 17 18
      0X00,
      0X00, //帧序号最大值 19 20
      0X00,
      0X00, //帧序号 21 22
      0X00,
      0X00, //校验和 23 24
      0XF8, //帧尾 25
    };
    int len = 26;
    int sum = 0;
    int dataLen = (data[17] << 8) + data[18];
    data[19] = (frame_Max >> 8) & 0XFF;
    data[20] = (frame_Max)&0XFF;
    data[21] = (frameNum >> 8) & 0XFF;
    data[22] = (frameNum)&0XFF;
    for (int i = 1; i < 19+dataLen; i++)
    {
        sum += data[i];
    }
    data[23] = (sum >> 8) & 0XFF;
    data[24] = (sum)&0XFF;
    //printf("len1 %d sum %d datalen %d\n",len,sum,dataLen);	

    //插值
    //0X88->0x88 0x00
    for (int i = 1; i < len - 1; i++)
    {
        if (data[i] == 0x88)
        {
            insert(0x00, &data[i + 1], len - 1 - i);
            len++;
        }
        
    }
	//printf("len2 %d\n",len);
    //插值+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //0XF7->0x88 0xF7
    //0XF8->0x88 0xF8
    for (int i = 1; i < len - 1; i++)
    {
        if (data[i] == 0xF7 && data[i - 1] != 0x88)
        {
            insert(0x88, &data[i ], len - i);
            len++;
        }
        else if (data[i] == 0xF8 && data[i - 1] != 0x88)
        {
            insert(0x88, &data[i ], len - i);
            len++;
        }
    }
	//printf("len3 %d\n",len);

    tty_write(tty_fd, (char*)data, len);
}

int frameSmoothControll(int now, int target)
{
    int absNumber = abs(target-now);
    int direct = target - now;

    //LogVerbose("absNumber = %d\n", absNumber);

    if(absNumber > THRESHOLD1)
    {
        if(direct > 0)
            return (now+DEGREE1);
        else
            return (now-DEGREE1);
    }
    else if(absNumber > THRESHOLD2)
    {
        if(direct > 0)
            return (now+DEGREE2);
        else
            return (now-DEGREE2);
    }
    else if(absNumber > THRESHOLD3)
    {
        if(direct > 0)
            return (now+DEGREE3);
        else
            return (now-DEGREE3);
    }
    else if(absNumber > THRESHOLD4)
    {
        if(direct > 0)
            return (now+DEGREE4);
        else
            return (now-DEGREE4);
    }
    else
        return now;

}
