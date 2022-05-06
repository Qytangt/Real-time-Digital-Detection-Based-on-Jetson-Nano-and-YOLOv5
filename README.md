# Real-time-Digital-Detection-Based-on-Jetson-Nano-and-YOLOv5
![uTools_1651399717330](https://user-images.githubusercontent.com/69594837/167097053-e8d4be70-7c3b-45d9-8603-b88745a43d62.png)
![pic1](https://user-images.githubusercontent.com/69594837/167097113-d0275748-4eca-463c-9c5d-edb7520646e1.png)

Real-time Digital Detection Based on Jetson Nano and YOLOv5
The application of intelligent robot is more and more widely today. Machine vision is an important part of intelligent robot. Digital plays a very important role in our life, so digital recognition has become an important field of machine vision. At present, most of the digital recognition systems are based on PC-Windos system. However, in intelligent robots, due to the limitation of calculation and storage space, the reasoning speed of the current large-scale and high-precision target detection model is low. Using Ultralytics ' YOLOv5 algorithm, a digital detector is constructed and deployed to embedded devices. The experimental results show that the proposed method uses TensorRT to accelerate the detection accuracy and speed on NVIDIA Jetson Nano embedded device, which is consistent with that on PC-Windos system. For 224 × 224 images, the detection speed of 25.72 ms per frame is realized.

基于Jetson Nano和YOLO v5的实时数字识别
在智能机器人的应用越来越广泛的今天。而机器视觉是智能机器人的重要组成部分。数字在我们的生活中扮演着非常重要的角色[1]，因此数字识别成为机器视觉的一个重要领域。现在大多是数字识别系统都是基于PC-Windos系统，而在智能机器人上，由于算力及存储空间的限制，当前的大型高精度目标检测模型的推理速度较低。采用 Ultralytics 的(对你只需看一次）YOLOv5算法，构造出一个数字检测器，并将其部署到嵌入式设备上。实验结果表明，本文提出方法在NVIDIA Jetson Nano嵌入式设备上，使用TensorRT加速在检测精度、速度与在PC-Windos系统上一致，对于 224×224 的图片，实现了 25.72ms 每帧的检测速度。

Based on the real-time digital recognition system of Jetson Nano and YOLO v5, the YOLO V5 algorithm has the advantages of faster recognition speed and higher accuracy than the traditional detection algorithm, so it can improve the efficiency of real-time digital recognition as a detection algorithm. This system can meet the daily work offline. The real environment is complex and changeable, there are still some problems to be solved, such as the influence of obstacles on the complex background, the detection of boxes is not ideal, and so on. At the same time, the introduction of the TensorRT acceleration module in Jetson Nano has greatly improved the digital speed compared to previous generation target recognition. Finally, the experimental results show that the proposed method is used on NVIDIA Jetson Nano embedded devices with a size of 224 × 224 single picture processing time is 25.72 Ms. Compared with other lightweight target detection models, it has faster inference speed and higher accuracy. The lightweight nature of the code allows the framework to be built on cheaper devices.

基于Jetson Nano和YOLO v5的实时数字识别系统，YOLO v5算法与传统检测算法相比，具有识别速度较快、精度较高等优点，故将该算法作为检测算法可以提高实时数字识别的效率。本系统能够满足离线脱机状况下的日常工作。现实环境复杂多变，依旧存在一部分问题有待解决，比如复杂背景下由于障碍物的影响，会对方框检测不太理想，诸如此类。同时，由于在Jetson Nano中TensorRT加速模块的引入，使得数字速度相比于前几代目标识别有大幅度的提升。最后的实验结果显示，本文提出方法在NVIDIA Jetson Nano嵌入式设备上,对尺寸为224×224的单图片处理时间为25.72ms。相比于其他轻量型目标检测模型有更快推理速度且有更高精度的优势。代码的轻量级使得该框架可以架构在更为便宜的设备上。

快速入门示例

安装
在 Python>=3.7.0环境中克隆 repo 并安装requirements.txt，包括 PyTorch>=1.7。

git clone https://github.com/Qytangt/Real-time-Digital-Detection-Based-on-Jetson-Nano-and-YOLOv5   #克隆

cd Real-time-Digital-Detection-Based-on-Jetson-Nano-and-YOLOv5

pip install -r requirements.txt   #安装

使用 detect.py 进行推理

detect.py在各种来源上运行推理，将结果保存到.runs/detect

python detect.py --source 0   # webcam                           
                          img.jpg   # image                           
                          vid.mp4   # video
                          path/   # directory
                          path/ * .jpg   # glob 

在Jetson-Nano开发板上使用 cap.py 进行实时数字识别

python cap.py

<img width="991" alt="uTools_1651221255890" src="https://user-images.githubusercontent.com/69594837/167096937-c781be46-d73a-471e-8d9d-5746b54dee01.png">

