---
title: Threshold
date: 2023-07-15 20:43:02
tags: opencv
categories: 项目
diytitle:
  enable: true
  leaveTitle: w(ﾟДﾟ)w 不要走！再看看嘛！
  backTitle: ♪(^∇^*)欢迎肥来！
cover: https://pic4.zhimg.com/80/v2-929070dd9619b4ebfec8a2436809a327_1440w.webp
---
# 说明
根据具体的目标调节对应的HSV  
**会用就好**
# 代码
```python
import cv2
import numpy as np
from torch import cartesian_prod

"""
    H:表示色度
    S:表示饱和度
    V:表示亮度
"""

Width = 640*0.5
Height = 480*0.5
cap = cv2.VideoCapture(1)
cap.set(3, Width)
cap.set(4, Height)
cap.set(10,150)

class Camra():
    def __init__(self) -> None:
        # 设置摄像头窗口尺寸
        cv2.namedWindow("HSV")
        cv2.resizeWindow("HSV",640,240)
        cv2.createTrackbar("HUE Min","HSV",0,179,self.empty)
        cv2.createTrackbar("SAT Min","HSV",0,255,self.empty)
        cv2.createTrackbar("VALUE Min","HSV",0,255,self.empty)
        cv2.createTrackbar("HUE Max","HSV",179,179,self.empty)
        cv2.createTrackbar("SAT Max","HSV",255,255,self.empty)
        cv2.createTrackbar("VALUE Max","HSV",255,255,self.empty)

    # 回调函数  可根据需求定义
    def empty(self,a):
        pass

    # 滑块的设置
    def slide(self):
        o, img = cap.read()
        imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("HUE Min","HSV")
        h_max = cv2.getTrackbarPos("HUE Max", "HSV")
        s_min = cv2.getTrackbarPos("SAT Min", "HSV")
        s_max = cv2.getTrackbarPos("SAT Max", "HSV")
        v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
        v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        # inrange去背景
        mask = cv2.inRange(imgHsv,lower,upper)
        # 通过与操作  去除不在阈值范围内的背景
        result = cv2.bitwise_and(img,img, mask = mask)
        # 拼接三张图一同显示
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        hStack = np.hstack([img,mask,result])
        cv2.imshow('Horizontal Stacking', hStack)


if __name__ == "__main__":
    cam = Camra()
    while True:
        cam.slide()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

```