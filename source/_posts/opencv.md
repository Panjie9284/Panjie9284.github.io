---
title: opencv
date: 2023-07-09 17:46:42
tags: opencv
categories: note
diytitle:
  enable: true
  leaveTitle: w(ﾟДﾟ)w 不要走！再看看嘛！
  backTitle: ♪(^∇^*)欢迎肥来！
cover: https://pic3.zhimg.com/80/v2-83746a8d97a0b33a8d90a395caadb5e2_720w.webp
---

# 1. opencv简介

## 1.1 环境搭建

先搭建新的虚拟环境  
```
conda create -n name python=3.x
```
name为环境名

3.x为指定python版本
在安装opencv之前需要安装nump，matplotlib
```
pip install opencv-python==3.4.2.17
```

建议安装opencv-python3.4.3版本以下，最新版本的一些算法可能需要付费

测试是否安装成功
```
import cv2
#调用cv2这个包
lena = cv2.imread("1.jpg")
#读取文件名为1.jpg的图片给lena
#cv2.imread("picturen_ame")
cv2.imshow("image",lena)
#显示图片
#picture_name = cv2.imshow(winname, img)
#winname：字符串，显示窗口的名称。
#img：所显示的 OpenCV 图像，nparray 多维数组。
cv2.waitKey(0)
#waitKey()–是在一个给定的时间内(单位ms)等待用户按键触发; 
#如果用户没有按下键,则继续等待 (循环)
#常见 : 设置 waitKey(0) , 则表示程序会无限制的等待用户的按键事件
#一般在 imgshow 的时候 , 如果设置 waitKey(0) , 代表按任意键继续
```
如果需要使用SIFT和SURF等进行特征提取时，还需要安装：
```
pip install opencv-contrib-python==3.4.2.17
#版本号最好保持一致
```
## 1.2 opencv的主要模块
**·core模块**实现了最核心的数据结构及其基本运算，如绘图函数、数组操作相关函数等。

**highgui模块**实现了视频与图像的读取、实示、存储等接口。

**imgproc模块**实现了图像处理的基本方法，包括图像滤波、图像几何变换、平滑、阈值分割、形态学处理、边缘检测、目标检测、运动分析和对象跟踪等。

**features2d**：图像特征及特征匹配

还有更多高层次模块，可以再去单独学习
# 2. opencv基本操作

1.图像的io操作

    1.API
    ```python
    cv.imread()
    ```
    ·cv.IMREAD*COLOR:彩色模式加载，默认参数
    ·cv.IMREAD*GRAYSCALE:灰度
    ·cv.IMREAD_UNCHANGED
    可以使用1、0、-1来替代这三个标志
    ```python
    import numpy as np
    import cv2 as cv
    #以灰度图的形式读取图像
    img = cv.imread("messi5.jpg",0)
    ```
2.显示图像  

1.API
```python
cv.imshow()
```
    ·显示图像窗口的名称，以字符串类型表示
    ·要加载的图片
```python
#opencv中显示
cv.imshow('image',img)
cv.waitKey(0)
#matplotlib中展示
plt.imshow(img[:,:,::-1])
#-1表示通道反转cv是BGR而plt是RGB
```

3.保存图像

```python
cv.imwrite("image/dilireba.png",img)

```

4.创建图片
   
```python
img = np.zeros((512,512,3),np.uint8)
```

5.绘制图形

```python
cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.circle(img,(256,256),60,(0,0,255),-1)
cv.rectangle(img,(100,100),(400,400),(0,255,0),5)
cv.putText(img,"ABC",(100,150),cv.FONT_HERSHEY_COMPLEX,5,(255,255,255),3)
```
6.获取和修改某个像素点

```python
px = img[100,100]
blue = img[100,100,0]
#仅获取蓝色通道的强度值
img[100,100] = [255,255,255]
#修改某位置的像素值
```

8.获取图像数据

img.shape 形状

img.size图像大小

img.dtype数据类型

9.图像通道拆分与合并

```python
b,g,r = cv.split(dili)

img2 = cv.merge((b,g,r))
```

10.色彩空间的改变

```python
gray = cv.cvtColor(dili,cv.COLOR_BGR2GRAY)
```
    ·intput_image:进行色彩空间转换的图像
    ·flag:转换类型
        ·cv.COLOP_BGR2GRAY:BGR->GRAY
        ·cv.COLOP_BGR2HSV:BGR->HSV

11.图像加法

```python
image3 = cv.add(image1,image2)
#cv加法
image4 = image1 + image2
#直接相加
```

cv相加完大于255直接为255
直接加法加完后还会取模
# 3. opencv的图像处理
## 3.1 几何变换 
1.图像缩放
```python
# 绝对尺寸
rows,cols = kids.shape[:2]
```

2.图像平移
```python
M = np.float32([[1,0,100],[0,1,50]])
res2 = cv.warpAffine(kids,M,(2*cols,2*rows))
```
参数  
    ·第一个图片名称  
    ·移动矩阵M  
    ·图像的大小  

3.图像的旋转
```python
M = cv.getRotationMatrix2D((cols/2,rows/2),45,0.5)
#getRotationMatrix2D(center, angle, scale)
res3 = cv.warpAffine(kids,M,(cols,rows))
```
参数
    ·center:图像的旋转中心  
    ·angle: 旋转角度  
    ·scale :一个各向同性的比例因子，根据提供的值将图像向上或向下缩放  
返回：
    ·M:旋转矩阵  
    调用cv.warpAffine完成图像旋转  

4.仿射变换

定义:是对图像的缩放，旋转，翻转和平移等操作的组合

仿射变换矩阵是 2 * 3 的矩阵

$A$ = 2 * 2，$B$ = 2 * 1  
```python
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[100,100],[200,50],[100,250]])
T = cv.getPerspectiveTransform(pst1,pst2)
res5 = cv.warpPerspective(kids,T,(cols,rows))
```
参数  
    ·  
    ·  
    ·  
数学理解比较重要，先跳过

5.透射变换

定义:利用投射中心、像点、目标点三点共线的条件，获得一个新的图形
```python
pst1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pst2 = np.float32([[100,145],[300,100],[80,290],[310,300]])
T = cv.getPerspectiveTransform(pst1,pst2)
res5 = cv.warpPerspective(kids,T,(cols,rows))
```

6.图像金字塔

定义:图像多尺度表达的一种，用于图像分割，以多分辨率来解释图像的有效但概念简单的结构

API
```python
cv.pyrUp(img)
#对图像进行上采样
cv.pyrDown(img)
#对图像进行下采样
```
## 3.2 形态学操作

1.连通性    
**概念1**  
    ·每个像素周围有8个相连，掌握概念，4邻接，D邻接，8邻接。  
**概念2**  
    ·两个像素连通的必要条件  
        是否相连  
        是否满足特定的相似性准则  
    ·4联通  
    ·8联通  
    ·m联通

2.腐蚀和膨胀  
腐蚀的作用:消除物体边界点，使目标缩小，可以消除小于结构元素的噪声点

具体操作过程，将要操作的图像的每个像素点依次进行操作，与核结构进行"与"操作，有0就为0，全1才为1。

API
```python
kenel = np.ones((5,5),np.uint8)
# 创建核结构
cv.erode(img,kernel,iterations)
```

参数  
    ·img:要处理的图像  
    ·kernel:核结构  
    ·iterations:腐蚀次数，默认为1 

膨胀的作用:使目标增大，可以添补目标的孔洞。

API
```python
kenel = np.ones((5,5),np.uint8)
# 创建核结构
cv.dilate(img,kernel,iterations)
```

参数  
    ·img:要处理的图像  
    ·kernel:核结构  
    ·iterations:膨胀次数，默认为1 

3.开闭运算（均不可逆）  
开运算:先腐蚀后膨胀  
    作用:分离物体，消除小区域。  
    特点：消除噪点，去除小的干扰块，而不影响原来的图像  
闭运算:先膨胀后腐蚀  
    作用:消除“闭合”物体里面的孔洞  
    特点:可以填充闭合区域  
API 
```python
kenel = np.ones((5,5),np.uint8)
# 创建核结构
cv.morphologyEx(img,op,kernel)
```

参数  
    ·img:要处理的图像  
    ·op:处理方式：  
        开运算，设为cv.MORPH_OPEN;
        闭运算，设为cv.MORPH_CLOSE  
    ·kernel:核结构  

4.礼貌和黑帽  
礼貌:开运算与原图的差。  
作用:分离比邻近点亮一些的斑块。在有大背景时，微小物品有规律时，可使用顶帽运算进行背景提取  
特点:突出了比原图轮廓周围的区域更亮的区域，与选择的核的大小相关  
黑帽:闭运算与原图的差。  
作用:分离比邻近点暗一些的斑块  
特点:突出了比原图轮廓周围的区域更暗的区域，与选择的核的大小相关  
API 
```python
kenel = np.ones((5,5),np.uint8)
# 创建核结构
cv.morphologyEx(img,op,kernel)
```
参数  
    ·img:要处理的图像  
    ·op:处理方式：  
        开运算，设为cv.MORPH_OPEN;  
        闭运算，设为cv.MORPH_CLOSE;  
        礼貌运算，设为cv.MORPH_TOPHAT;  
        黑帽运算，设为cv.MORPH_BLACKHAT;  
    ·kernel:核结构  
## 3.3 图像平滑

1.图像噪声  
理解概念:  
    椒盐噪声（脉冲噪声）:随机出现黑色白色斑点  
    高斯噪声:分布满足高斯公式，拥有各个灰度值的噪声  

2.图像平滑  
使用滤波来去除图像中的噪声

3.均值滤波  
原理:需再去理解  
优点:算法简单，计算速度快  
缺点:去噪的同时去除了很多细节将图像变得模糊  
API
```python
cv.blur(src, ksize, anchor, borderType)
```
参数  
    ·src:输入图像  
    ·ksize:卷积核的大小
    ·anchor:默认值（-1，-1），表示核中心
    ·borderType:边界类型

4.高斯滤波  
跟据高斯公式，在像素周围得到一个加权矩阵。
作用:专门处理高斯噪声    
API
```python
    cv.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
```
参数  
    ·src:输入图像  
    ·ksize:高斯卷积核的大小，宽度和高度都应为奇数，且可以不同  
    ·sigmaX:水平方向的标准差
    ·sigmaY:垂直方向的标准差
    ·borderType:填充边界类型

5.中值滤波  
利用像素点邻域灰度值的中值来代替该点的灰度值  
作用:对椒盐噪声尤其有用  
API  
```python
cv.medianBlur(src,ksize)
```
参数  
    ·src:输入图像  
    ·ksize:卷积核的大小  
## 3.4 直方图

1.一些**常见术语**  
·dims:需要统计特征数目  
·bins:每个特征空间子区段的数目，可理解为直条或组距  
·range:需要统计特征的取值范围  

2.直方图的**意义**  
·直方图是图像中像素强度分布的图形表达方式  
·他它统计了每一个强度值所具有的像素个数  
·不同图像的直方图可能是相同的  

3.直方图的计算和绘制  
API  
```python
cv2.calcHis(images,channels,mask,histSize,,ranges[,hist[,accumulate]])
```
参数  
·images:原图像  
·channels:灰色图为0；彩色图中0，1，2，分别对应B，G，R  
·mask:掩模图像。要统计整个图设为None。如果想统计某一部分，需自己制作一个掩模图像
·histSize:BIN的数目。应用中括号括起来  
·ranges:像素值范围，通常为[0,256]  
**不是很懂**

4.掩膜的应用  
定义:使用选定的图像、图形或物体，对要处理的图像进行遮掩，来控制图像处理的区域  
使用二维数组矩阵数组进行掩膜。掩膜由0和1组成，1会被处理，0不会被处理  
主要用途:
·提取感兴趣的区域  
·屏蔽作用  
·结构特征提取  
·特殊形状图形制作  
使用cv.calcHist()来查找完整图像的直方图。  
```python
# 创建掩膜
mask = np.zeros(img.shape[:2],np.uint8)
mask[400:650,200:500] = 1
#掩膜的图像
mask_img = cv.bitwise_and(img,img,mask=mask)
```

5.直方图均衡化  
把原始图像的灰度直方图从比较集中的某个灰度区域变成更广泛灰度范围内的分布。  
运用在曝光过度和曝光不足，x光上  
API
```python
dst = cv.equalizeHist(img)
```
参数  
·img:灰度图像  
返回  
·dst:均衡化后的结果  

6.自适应的直方图均衡化

增加对比度限制。    
API
```python
cv.createCLAHE(clipLimit, tileGridSize)
```
参数  
·clipLimit:对比度限制，默认是40  
·tileGridSize:分块的大小，默认为 8 * 8  

7.边缘检测  
作用:大幅度的减少了数据量，并剔除了可以认为不相关的信息，保留了图像的结构属性。  
有许多方法用于边缘检测，它们绝大大部分可以划为两类：基于搜索和基于零穿越。  
## 3.5 边缘检测

1.Sobel检测算子  
**原理再看**  
API
```python
Sobel_x_or_y = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
```
参数  
·src:传入的图像  
·ddepth:图像深度  
·dx和dy:指求导的阶数，0表示这个方向上没有求导，取值为0、1  
·ksize:是Sobel算子的大小，即卷积核的大小，必须为奇数1、3、5、7，默认为3。  
·scale:缩放导数的比例常数，默认情况为没有伸缩系数。  
·borderType:图像边界的模式，默认值为cv2.BORDER_DEFAULT  

2.Schaar算子  
```python
x =cv.Sobel(img,cv.CV_16S,1,0,ksize=-1)
y = cv.Sobel(img,cv.CV_16S,0,1,ksize=-1)
absx = cv.convertScaleAbs(x)
absy = cv.convertScaleAbs(y)
res = cv.addWeighted(absx,0.5,absy,0.5,0)
plt.imshow(res,cmap=plt.cm.gray)
```
3.Laplacian算子  
```python
res = cv.Laplacian(img,cv.CV_16S)
res = cv.convertScaleAbs(res)
plt.imshow(res,cmap=plt.cm.gray)
```
4.Canny边缘检测  
API
```python
canny = cv2.Canny(img, threshold1, threshold2)
```
参数  
·image:灰度图  
·threshold1:minval，较小的阈值将间断的边缘连接起来  
·threshold2:maxval，较大的阈值检测图像中明显的边缘  
## 3.6 模板匹配和霍夫变换

1.模板匹配  
原理:在所给图片中查找和模板最相似的区域。  
API:
```python
res = cv.matchTemplate(img, template, method)
cv.minMaxLoc()
#搜索最匹配的位置
```
参数  
·img:要进行模板匹配的图像  
·Template:模板  
·method:实现模板匹配的算法，主要有:  
平方差匹配CV_TM_SQDIFF，最好匹配是0，匹配越差，值越大  
相关匹配CV_TM_CCORR，值越大效果越好  
相关系数匹配CV_TM_CCOEFF，1表示完美匹配，-1表示最差匹配  

2.霍夫变换  
直线检测  
```python
cv.HoughLines(img, rho, theta, threshold)
```
参数  
·  img:检测的图像，要求是二值化的图像  
·  rho、theta:两个参数的精度  
·  threshold:阈值，只有累加器中的值高于该阈值才被认为是直线  
霍夫圆检测  
API  
```python
ircles = cv.HonghCircles(image, method, dp, minDist, param1=100, param2=100, minRadius=0, maxRadius=0)
```
参数(圆心和半径均需要为int)  
·image:输入图像应输入灰度图像  
·method:使用霍夫变换圆检测的算法，它的参数是CV_HOUGH_GRADIENT  
·dp:霍夫空间的分辨率，dp=1时大小一致，dp=2时为输入空间的一半
·minDist:为圆心之间最小距离，如果两个圆心之间距离小于该值，则认为它们是同一个圆心  
·param1:边缘检测时使用Canny算子的高阈值，低阈值是高阈值的一半  
·param2:检测圆心和确定半径时所共有的阈值  
·minRadius和maxRadius:为检测到的圆半径的最小值和最大值  
返回  
·ciecles:输出圆向量，包括三个浮点型的元素--圆心横坐标，圆心纵坐标和圆半径  
# 4. 角特征

1.图像特征  
类比拼图  
要有区分性，容易被比较。一般认为角点，斑点等是比较好的图像特征  

2.Harris角点检测
**原理：再去看**  
API
```python
dst = cv.cornerHarris(src, blockSize, ksize, k)
```
参数  
·img:数据类型为float32的输入图像  
·blockSize:较大检测中要考虑的邻域的大小  
·ksize:sobel求导使用的核大小  
·k:角点检测方程中的自由参数，取值参数为[0.04,0.06]  

3.Shi-Tomasi角点检测
API
```python
corners = cv2.goodFeaturesToTrack(image, maxcorners, qualityLevel, minDistance)
```
参数  
·Image:输入灰度图像  
·maxCorners:获取角点数的数目  
·qualityLevel:该参数指出最低可接受的角点质量水平，在0-1之间  
·minDistance:角点之间最小的欧式距离，避免得到相邻特征点  
返回  
·Corners:搜索到的角点，在这里所有低于质量水平的角点被排除掉，然后把合格的角点按质量排序，然后将质量较好的角点附近（小于最小欧式距离）的角点删掉，最后找到maxCorners个角点返回  

4.SIFT  
**原理：先跳过了**  
实例化sift  
```python
sift = cv.xfeatures2d.SIFT_creat()
```
利用sift。detectAndCompute()检测关键点并计算
```python
kp,des = sift.detectAndCompute(gray,None)
```
参数  
·gary:进行关键点检测的图像，注意是灰度图像
返回  
·kp:关键点信息，包括位置，尺度，方向信息  
·des:关键点描述符，每个关键点对应128个梯度信息的特征向量  
将关键点检测结果绘制在图像上  
```python
cv.drawKeypoints(image, keypoints, outputimage, color, flags)
```
参数  
·image:原始图像  
·keypoints:关键点信息，将其绘制在图像上  
·outputimage:输出图片，可以是原始图像  
·color:颜色设置，通过修改(b,g,r)的值，更改画笔的颜色  
·flags:绘图的标识设置  
(1)cv2.DRAW_WATCHES_FLAGS_DEFAULT:创建输出图像矩阵，使用现存的输出图像绘制匹配对和特征点，对每一个关键点只绘制中间点  
(2)cv2.DRAW_WATCHES_FLAGS_DRAW_OVER_OUTIMG:不创建输出图像矩阵，而是在输出图像上绘制匹配对  
(3)cv2.DRAW_WATCHES_FLAGS_DRAW_RICH_KEYPOINTS:对每一个特征点绘制带大小和方向的关键点图形  
(4)cv2.DRAW_WATCHES_FLAGS_DRAW_SINGLE_POINTS:单点的特征点不被绘制  

5.FAST算法  
实例化fast  
```python
fast = cv.FastFeatureDetector_create(threshold, nonmaxSuppression)
```
参数  
·threshold:阈值t，有默认值10  
·nonmaxSuppression:是否进行非极大值抑制，默认值True  
返回  
·Fast:创建的FastFeatureDetector对象  
利用fast.detect检测关键点，没有对应关键点描述  
```
kp = fast.detect(grayImg, None)
```
参数  
·gray:进行关键点检测的图像，注意是灰度图像  
返回  
·kp:关键点信息，包括位置，尺度，方向信息  
将关键点检测结果绘制在图像上，与SIFT一样  
```python
cv.drawKeypoints(image, keypoints, outputimage, color, flags)
```
参数  
·image:原始图像  
·keypoints:关键点信息，将其绘制在图像上  
·outputimage:输出图片，可以是原始图像  
·color:颜色设置，通过修改(b,g,r)的值，更改画笔的颜色  
·flags:绘图的标识设置  
(1)cv2.DRAW_WATCHES_FLAGS_DEFAULT:创建输出图像矩阵，使用现存的输出图像绘制匹配对和特征点，对每一个关键点只绘制中间点  
(2)cv2.DRAW_WATCHES_FLAGS_DRAW_OVER_OUTIMG:不创建输出图像矩阵，而是在输出图像上绘制匹配对  
(3)cv2.DRAW_WATCHES_FLAGS_DRAW_RICH_KEYPOINTS:对每一个特征点绘制带大小和方向的关键点图形  
(4)cv2.DRAW_WATCHES_FLAGS_DRAW_SINGLE_POINTS:单点的特征点不被绘制  

6.ORB算法  
实例化ORB  
```python
orb = cv.xfeatures2d,orb_create(nfeatures)
```
参数  
·nfeatures:特征点的最大数量  
利用orb.detectAndCompute()检测关键点并计算  
```python
kp,des = orb.detectAndCompute(gray, None)
```
参数  
·gray:进行关键点检测的图像，注意特别是灰度图像  
返回  
·kp:关键点信息，包括位置，尺度，方向信息  
·des:关键点描述符，每个关键点BRIEF特征向量，二进制字符串  
将关键点检测结果绘制在图像上
```python
cv.drawKeypoints(image, keypoints, outputimage, color, flags)
```
# 5. 视频操作
1.视频读写  
```python
cap = cv.VidoCapture(filepath)
```
参数  
·filepath:视频文件路径  
获取视频的某些属性  
```python
retval = cap.get(propId)
```
参数  
propld表  
param	英文	define  
cv2.VideoCapture.get(0)	CV_CAP_PROP_POS_MSEC	视频文件的当前位置（播放）以毫秒为单位  
cv2.VideoCapture.get(1)	CV_CAP_PROP_POS_FRAMES	基于以0开始的被捕获或解码的帧索引  
cv2.VideoCapture.get(2)	CV_CAP_PROP_POS_AVI_RATIO	视频文件的相对位置（播放）：0=电影开始，1=影片的结尾  
cv2.VideoCapture.get(3)	CV_CAP_PROP_FRAME_WIDTH	在视频流的帧的宽度  
cv2.VideoCapture.get(4)	CV_CAP_PROP_FRAME_HEIGHT	在视频流的帧的高度  
cv2.VideoCapture.get(5)	CV_CAP_PROP_FPS	帧速率  
cv2.VideoCapture.get(6)	CV_CAP_PROP_FOURCC	编解码的4字-字符代码  
cv2.VideoCapture.get(7)	CV_CAP_PROP_FRAME_COUNT	视频文件中的帧数  
cv2.VideoCapture.get(8)	CV_CAP_PROP_FORMAT	返回对象的格式  
cv2.VideoCapture.get(9)	CV_CAP_PROP_MODE	返回后端特定的值，该值指示当前捕获模式  
cv2.VideoCapture.get(10)	CV_CAP_PROP_BRIGHTNESS	图像的亮度(仅适用于照相机)  
cv2.VideoCapture.get(11)	CV_CAP_PROP_CONTRAST	图像的对比度(仅适用于照相机)  
cv2.VideoCapture.get(12)	CV_CAP_PROP_SATURATION	图像的饱和度(仅适用于照相机)  
cv2.VideoCapture.get(13)	CV_CAP_PROP_HUE	色调图像(仅适用于照相机)  
cv2.VideoCapture.get(14)	CV_CAP_PROP_GAIN	图像增益(仅适用于照相机)（Gain在摄影中表示白平衡提升）  
cv2.VideoCapture.get(15)	CV_CAP_PROP_EXPOSURE	曝光(仅适用于照相机)  
cv2.VideoCapture.get(16)	CV_CAP_PROP_CONVERT_RGB	指示是否应将图像转换为RGB布尔标志  
cv2.VideoCapture.get(17)	CV_CAP_PROP_WHITE_BALANCE	× 暂时不支持  
cv2.VideoCapture.get(18)	CV_CAP_PROP_RECTIFICATION	立体摄像机的矫正标注（目前只有DC1394 v.2.x后端支持这个功能）  
————————————————

2.判断是否读取成功
```python
isornot = cap.isOpened()
#返回值为true或false
```

3.某一帧读取  
```python
ret, frame = cap.read()
#ret为是否受到为true或false
#frame为接收的图像
```

4.修改视频属性
```python
cap.set(prople, value)
#属性的索引和需要更改后的属性
```

5.保存视频  
```python
out = cv2.VideoWriter(filename,fourcc, fps, frameSize)
```
参数  
·filename：视频保存的位置  
·fourcc：指定视频编解码器的4字节代码  
·fps：帧率  
·frameSize：帧大小  

6.设置视频的编解码器  
```python
retval = cv2.VideoWriter_fourcc( c1, c2, c3, c4 )

```
参数  
c1,c2,c3,c4: 是视频编解码器的4字节代码，在fourcc.org中找到可用代码列表，与平台紧密相关，常用的有：  
在Windows中：DIVX（.avi）  
在OS中：MJPG（.mp4），DIVX（.avi），X264（.mkv）  

7.视频追踪Meanshift  
API
```python
cv.meanShift(probImage, window, criteria)
```
参数   
probImage: ROI区域，即目标的直方图的反向投影  
window： 初始搜索窗口，就是定义ROI的rect  
criteria: 确定窗口搜索停止的准则，主要有迭代次数达到设置的最大值，窗口中心的漂移值大于某个设定的限值等  

8.Camshift  
```python
 # 4.4 进行meanshift追踪
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # 4.5 将追踪的位置绘制在视频上，并进行显示
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
```
```python
  #进行camshift追踪
    ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # 绘制追踪结果
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
```
# 6. 案例