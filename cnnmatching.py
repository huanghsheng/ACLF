import argparse
import cv2
import numpy as np
import imageio
import plotmatch
from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
import time
import os
from skimage import measure
from skimage import transform
# os.chdir(r'/home/xujiajun/allAlgorithm/CMM-NET')
#time count
import time
_RESIDUAL_THRESHOLD = 30
#Test1nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
# imgfile1 = 'df-ms-data/1/df-googleearth-1k-20091227.jpg'
# imgfile2 = 'df-ms-data/1/df-googleearth-1k-20181029.jpg'
# 方式1
# imgfile2 = 'df-ms-data/2/I1.jpg'
# imgfile1 = 'df-ms-data/2/V1.jpg'
# 方式2

'''第一部分 读入图像'''

imgfile1 = './baiyun_gray_size/137_T3.png'
imgfile2 = './baiyun_gray_size/137_W3.png'
# matrix11='C7_T.mat'1# matrix22='C7_W.mat'
h=30


# 方式3
# imgfile2 = cv2.imwrite('images/I2.jpg')
# imgfile2 = cv2.cvtColor(imgfile2, cv2.COLOR_BGR2GRAY)
# imgfile1 = cv2.imwrite('images/W2.jpg')
# imgfile1 = cv2.cvtColor(imgfile1, cv2.COLOR_BGR2GRAY)

# imgfile1 = 'df-ms-data/1/df-uav-sar-500.jpg'



# start=time.time()
# read left image
image1 = imageio.imread(imgfile1)
image2 = imageio.imread(imgfile2)

# print('read image time is %6.3f' % (time.perf_counter() - start))
start = time.time()


# ract(image1,  nfeatures = -1)
# 分别为关键点，得分，描述符,行数是一样的
# kps_left.shape
# (280, 3)
# sco_left.shape
# (280,)
# des_left.shape
# (280, 512)
kps_left, sco_left, des_left = cnn_feature_extract(image1,  nfeatures = -1)
kps_right, sco_right, des_right = cnn_feature_extract(image2,  nfeatures = -1)

print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))
# start = time.perf_counter()


'''第二部分 Flann特征匹配,产生最终的匹配点'''
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=40)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_left, des_right, k=2)



goodMatch = []
locations_1_to_use =[]
se = []
locations_2_to_use = []

# 匹配对筛选
min_dist = 1000
max_dist = 0
disdif_avg = 0
# 统计平均距离差
# for m, n in matches:
#     print(m.distance,n.distance)
# n.distance 都会比m.distance 大
for m, n in matches:
    disdif_avg += n.distance - m.distance
disdif_avg = disdif_avg / len(matches)
# i=0
for m, n in matches:
    #自适应阈值
    if n.distance > m.distance + disdif_avg:
        goodMatch.append(m)
        p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
        p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
        locations_1_to_use.append([p1.pt[0], p1.pt[1]])
        locations_2_to_use.append([p2.pt[0], p2.pt[1]])
    # i=i+1
    # if i==100:
    #     break
#goodMatch = sorted(goodMatch, key=lambda x: x.distance)
print('match num is %d' % len(goodMatch))
locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)
'''第三部分 对产生的匹配点进行初步的处理，主要是这部分代码，要调好'''
# 求出所有的角度和长度

import math

def calculate_angle(x1, y1, x2, y2):
    # 计算两点的向量
    dx = x2 - x1
    dy = y2 - y1

    # 计算角度（弧度）
    # 使用atan2函数，它返回的是从正x轴逆时针旋转到点的角度
    angle_rad = math.atan2(dy, dx)

    # 转换为度
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def calculate_distance(x1, y1, x2, y2):
    # 使用欧几里得距离公式计算两点之间的距离
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# 示例
# x1, y1 = 1, 1
# x2, y2 = 4, 5
# angle = calculate_angle(x1, y1, x2, y2)
# print(f"The angle between the points is {angle} degrees")

# '''第三部分 自己加的，筛选出符合条件的a和l'''
# 求a和l的平均值
aall = []
lall = []
max_a=calculate_angle(locations_1_to_use[1, 0], locations_1_to_use[1, 1],  (locations_2_to_use[1, 0]+650), locations_2_to_use[1, 1])
min_a=calculate_angle(locations_1_to_use[1, 0], locations_1_to_use[1, 1],  (locations_2_to_use[1, 0]+650), locations_2_to_use[1, 1])
max_l=calculate_distance(locations_1_to_use[1, 0], locations_1_to_use[1, 1], (locations_2_to_use[1, 0]+650), locations_2_to_use[1, 1])
min_l=calculate_distance(locations_1_to_use[1, 0], locations_1_to_use[1, 1], (locations_2_to_use[1, 0]+650), locations_2_to_use[1, 1])

for i in range(locations_1_to_use.shape[0]):
    # a = np.arctan((locations_2_to_use[i, 1] - locations_1_to_use[i, 1]) / (locations_2_to_use[i, 0] - locations_1_to_use[i, 0]))
    a = calculate_angle(locations_1_to_use[i, 0], locations_1_to_use[i, 1],  (locations_2_to_use[i, 0]+650), locations_2_to_use[i, 1])
    if a > max_a:
        max_a = a
    if a < min_a:
        min_a = a
    # l = ((locations_2_to_use[i, 1] - locations_1_to_use[i, 1]) * (locations_2_to_use[i, 1] - locations_1_to_use[i, 1]) + (locations_2_to_use[i, 0] - locations_1_to_use[i, 0]) * (locations_2_to_use[i, 0] - locations_1_to_use[i, 0]) ) ** 0.5
    l = calculate_distance(locations_1_to_use[i, 0], locations_1_to_use[i, 1], (locations_2_to_use[i, 0]+650), locations_2_to_use[i, 1])
    if l > max_l:
        max_l = l
    if l < min_l:
        min_l = l
    aall.append(a)
    lall.append(l)

# 可以先看长度，在考虑角度问题

meanamatrix=np.mean(aall)
meanlmatrix=np.mean(lall)
# print(amatrix)
print(meanamatrix)
# print(meanamatrix)
print(meanlmatrix)
# 主要保证配准不会失败

d=(max_a-min_a)*(1/3)
l=(max_l-min_l)*(1/3)
# d=0
# l=0

# m=0
# # 就是求符合a并且符合l的连接线的个数，然后把他们的点给赋给矩阵
# for i in range(locations_1_to_use.shape[0]):
#     a = calculate_angle(locations_1_to_use[i, 0], locations_1_to_use[i, 1],  (locations_2_to_use[i, 0]+660), locations_2_to_use[i, 1])
#     l = calculate_distance(locations_1_to_use[i, 0], locations_1_to_use[i, 1], (locations_2_to_use[i, 0]+660), locations_2_to_use[i, 1])
#     if ((meanamatrix-15)<=a<=(meanamatrix+15)) and ((meanlmatrix-350)<=l<=(meanlmatrix+350)):
#         # if (meanlmatrix - d) <= l <= (meanlmatrix + d):
#         m+=1
# print(m)
# 后面不是求符合a并且符合l的连接线的个数，而是直接筛选出合适的连接线，然后把他们的相应的匹配点对给赋给矩阵



j=0
matrix3=np.zeros((locations_1_to_use.shape[0],2))
matrix4=np.zeros((locations_1_to_use.shape[0],2))
for i in range(locations_1_to_use.shape[0]):
    m = calculate_angle(locations_1_to_use[i, 0], locations_1_to_use[i, 1],  (locations_2_to_use[i, 0]+660), locations_2_to_use[i, 1])
    n = calculate_distance(locations_1_to_use[i, 0], locations_1_to_use[i, 1], (locations_2_to_use[i, 0]+660), locations_2_to_use[i, 1])
    if ((min_a+abs(d)-1)<=m<=(max_a-abs(d)+1)) and ((min_l+abs(l)-1)<=n<=(max_l-abs(l)+1)):
        # if (meanlmatrix - d) <= l <= (meanlmatrix + d):
        matrix3[j, 0] = locations_1_to_use[i, 0]         # 有问题
        matrix3[j, 1] = locations_1_to_use[i, 1]
        matrix4[j, 0] = locations_2_to_use[i, 0]
        matrix4[j, 1] = locations_2_to_use[i, 1]
        j=j+1

matrix3 = np.delete(matrix3, np.where(~matrix3.any(axis=1))[0], axis=0)
matrix4 = np.delete(matrix4, np.where(~matrix4.any(axis=1))[0], axis=0)
'''第四部分 使用ransac来进行筛选匹配点 '''
# Perform geometric verification using RANSAC.
_, inliers = measure.ransac((matrix3, matrix4),
                          transform.AffineTransform,
                          min_samples=3,
                          residual_threshold=_RESIDUAL_THRESHOLD,
                          max_trials=1000)

print('Found %d inliers' % sum(inliers))

inlier_idxs = np.nonzero(inliers)[0]
# 最终匹配结果
matches = np.column_stack((inlier_idxs, inlier_idxs))
# print('whole time is %6.3f' % (time.perf_counter() - start0))

'''这部分代码是为了保存最后生成的配准点的，没其他作用'''
j=0
i=0
n=0
m=inlier_idxs.shape[0]
matrix1=np.zeros((m,2))
matrix2=np.zeros((m,2))
for i in range(inlier_idxs.shape[0]):
    n=inlier_idxs[i:i+1]
    matrix1[j, 0] = matrix3[n, 0]
    matrix1[j, 1] = matrix3[n, 1]
    matrix2[j, 0] = matrix4[n, 0]
    matrix2[j, 1] = matrix4[n, 1]
    j=j+1
num_rows = matrix1.shape[0]
row_indices = np.random.choice(num_rows, size=h, replace=False)
random_rows1 = matrix1[row_indices, :]
random_rows2 = matrix2[row_indices, :]


'''第五部分 保存矩阵'''
from scipy.io import savemat
# 将矩阵保存为 .mat 文件
savemat('matrix1.mat', {'matrix': random_rows1})
savemat('matrix2.mat', {'matrix': random_rows2})
print('导成.mat文件')

end=time.time()
print("匹配点匹配运行时间:%.4f秒" % (end-start))

# 顺便把RMSE和MSE指标给计算出来，免得还要去matlab 计算 时间的话是1.7768，RMSE是2.0752，RMSE是1.5044，要比这个好
# 以matlab 为准，python 算出来数值和matlab不一样

'''第六部分 显示图像，只是提取匹配点的话，下面的是全部可以注释掉的'''
# Visualize correspondences, and save to file.
#1 绘制匹配连线
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['figure.figsize'] = (4.0, 3.0) # 设置figure_size尺寸
_, ax = plt.subplots()


# 精细化匹配可以从这里加
plotmatch.plot_matches(
    ax,
    image1,
    image2,
    matrix3,
    matrix4,
    np.column_stack((inlier_idxs, inlier_idxs)),
    plot_matche_points = False,
    matchline = True,
    matchlinewidth = 0.7)
ax.axis('off')
ax.set_title('')
plt.show()
