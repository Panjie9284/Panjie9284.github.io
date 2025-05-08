---
title: demo1
date: 2023-07-15 20:24:37
tags: boom boom fly
categories: 项目
diytitle:
  enable: true
  leaveTitle: w(ﾟДﾟ)w 不要走！再看看嘛！
  backTitle: ♪(^∇^*)欢迎肥来！
cover: https://pic2.zhimg.com/80/v2-c7e1d8b6d113c79918e9e5037c7b033d_1440w.webp
---
## 功能说明
主要实现的功能为首先接收飞控mode2时，键盘输入两个指定的大概坐标，再接收到飞控传来的mode1，进入追色块模式返回色块中点（此处色块只包含红蓝两种颜色）

## 代码实现
```python
import cv2
import numpy as np
import os

from Communications import SelfSerial 
from SplitInt import get_high_low_data
from loguru  import logger

#引用一些包

os.system('echo rock | sudo -S chmod 777 /dev/ttyUSB0')

red_lower = np.array([0, 43, 46])
red_upper = np.array([10, 255, 255])
#红色的阈值
    
blue_lower = np.array([100, 43, 46])
blue_upper = np.array([124, 255, 255])
#蓝色的阈值

def find_largest_color_blob(image, color_lower, color_upper,min_area = 500):#自定义找图像中最大色块的函数

    # 转换颜色空间为HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建颜色阈值
    color_mask = cv2.inRange(hsv_image, color_lower, color_upper)
    
    # 执行形态学操作以消除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 寻找最大轮廓
    largest_contour = None
    largest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area and area > min_area:
            largest_area = area
            largest_contour = contour
    
    # 如果找到最大轮廓
    if largest_contour is not None:
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            # 计算中心点坐标
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return center_x, center_y

    return None
    #如果没有得到图像就返回None

cap = cv2.VideoCapture(1)
#开启摄像头

cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

self_serial = SelfSerial('/dev/ttyUSB0')


mode = 0
#mode初始化

logger.info("System Starting")
#程序开始时打印说明程序开始

while True:#一直执行
    ret, frame = cap.read()
    #读取摄像头

    if ret:
        
        mode = self_serial.uart_read_mode(mode)
        #从飞控处获得当前模式

        if mode == 99:
            pass
        if mode == 1:#mode 1追色块
            if frame is not None:
                try:#尝试寻找图像中色块
                    red_center = find_largest_color_blob(frame, red_lower, red_upper)
                    blue_center = find_largest_color_blob(frame, blue_lower, blue_upper)
                    #调用前面定义的图像最大色块函数
                    
                    # 在帧中绘制识别结果
                    if red_center is not None:
                        cv2.circle(frame, red_center, 5, (0, 0, 255), -1)
                        x = int(red_center[0])
                        y = int(red_center[1])
                        print(x, y)
                        msg = get_high_low_data(x) + get_high_low_data(y)
                    else:
                        if blue_center is not None:
                            cv2.circle(frame, blue_center, 5, (255, 0, 0), -1)
                            x = int(red_center[0])
                            y = int(red_center[1])
                            print(x, y)
                            msg = get_high_low_data(x) + get_high_low_data(y)
                except:#未找到色块时
                    x = 0
                    y = 0#确保有不会特别离谱的坐标值返回给飞控
                    print(x, y)
                    msg = get_high_low_data(x) + get_high_low_data(y)
                cv2.imshow('camera', frame)
                cv2.waitKey(1)                                  
                self_serial.uart_send_msg(0x01, msg)
        if mode == 2 :#mode2 输入指定坐标
            if frame is not None:
                print("请输入三个x坐标和y坐标")
                x1 = int(input())
                y1 = int(input())
                x2 = int(input())
                y2 = int(input())

                print(x1,y1)
                print(x2,y2)
                msg = get_high_low_data(x1) + get_high_low_data(y1) + get_high_low_data(x2) + get_high_low_data(y2)
                self_serial.uart_send_msg(0x02, msg) 
                mode = 99
                cv2.imshow('camera', frame)
                cv2.waitKey(1)
```