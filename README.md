
# ACLF

Pytorch implements visible light and thermal infrared image registration of rice phenotype unmanned aerial vehicles.
This study aims to establish a general framework for the application of registration technology in visible light and thermal infrared images of unmanned aerial vehicles, and proposes an adaptive connection line filtering algorithm for the registration of visible light and thermal infrared images of rice phenotype unmanned aerial vehicles.

## Data preparation
本研究的数据集实在自然无风的条件下采集的。

## Requirements
- Python3.8
- Pytorch 1.9.1
- matlab 2015b
- Cuda 11.1
- Ubuntu20.04 Linux
- CPU:Intel Xeon Silver 4210R processor×2
- 32 GB memory×8
- hard disk:RTX3090 24 GB graphics card×4

## Code Folder Description
myAlgorithm 自己的算法：

       各个 文件夹 介绍：
       lib 里面有卷积特征提取模块
       xiaorong 里面包含一到七文件夹，七个消融实验产生的粗匹配点
       shougong_baiyun_mat1 和 shougong_zengcheng_json 里面的json文件，分别是白云和增城标注的真实值，包含红外点，可见光点，真实值的单应矩阵，以及损失值


       各个 文件  介绍：
       matrix1.mat 和 matirx2.mat 文件 初步匹配后的匹配点矩阵文件
       text3.py 输入是手工标注的真实点(json文件)，还有精细匹配后的两对点(两个.mat矩阵文件)，输出是RMSE（输出1）。红外精细匹配点经过精细匹配单应矩阵，得到映射后的在可见光上的点对（输出2），
       然后再和手动标注的可见光图像点（输出3）对计算RMSE值。后面的RMSE和MSE是由输出2和输出3计算的


y_showResult 文件介绍：
       CMM_NET，GLAWpoints等文件，里面是手动标注的点对
       huitu_image.m，meanMAE.m，xiangxiantu.m等文件是用来跑实验定量和定性结果的
       y_mat_T和y__mat_W里面分别是是映射到可见光图像上的红外点对，和手动标注的可见光点对，实验结果是基于这些来跑的

cutImage 文件介绍：
       里面主要是手工裁剪和标注真实值并生成真实值相应文件的代码

       各个 文件夹 介绍：
       input 粗略裁剪的输入图
       output 输出图


       各个 文件介绍：
       test1.m和text4.m 精细手动裁剪图片并保存，网格化窗口显示结果
       test2.py 把手动标注的.mat文件生成json文件
       RMSEandMAE.py 计算RMSE和MAE
       Gray.m 批量灰度化，对比度增强和尺寸统一，可选全部或者其中的一项
       Gray_single 单张灰度化，其他同上

其他两个文件分别是农科院和增城拍的原始数据
