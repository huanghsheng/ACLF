import torch
from lib.model import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale
import scipy
import scipy.io
import scipy.misc
import numpy as np

use_cuda = torch.cuda.is_available()

# Creating CNN model
model = D2Net(  # 这个函数，待会儿研究下
    model_file="models/d2_ots.pth",
    use_relu=True,
    use_cuda=use_cuda
)
device = torch.device("cuda:0" if use_cuda else "cpu")

multiscale = True
max_edge = 2500
max_sum_edges = 5000
# de-net feature extract function
def cnn_feature_extract(image,scales=[.25, 0.50, 1.0], nfeatures = 1000):
    if len(image.shape) == 2:
        # 假设x为
        # array([[0.59632172, 0.24006924],
        #        [0.69303062, 0.87381667]])
        image = image[:, :, np.newaxis] # np.newaxis 用于增加维度
        # array([[[0.59632172],
        #         [0.24006924]],
        #        [[0.69303062]
        #         [0.87381667]]])
        image = np.repeat(image, 3, -1)
    # array([[[0.59632172, 0.59632172, 0.59632172],
    #         [0.24006924, 0.24006924, 0.24006924]],
    #        [[0.69303062, 0.69303062, 0.69303062],
    #         [0.87381667, 0.87381667, 0.87381667]]])

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > max_edge: # max(x.shape) 的意思是取第一个，如shape为（2，3，4） 则结果为2
        resized_image = scipy.misc.imresize(  # scipy 这个版本可能不支持imresize,如果resized_image 大于 2500， 可能就会报错
            resized_image,
            max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > max_sum_edges:  # 例如 x.shape=(2,3,4) x.shape(: 2) 就是取图像的长和宽，也就是第一个和第二个；x.shape(: 3)就是取图像的通道数，也就是第三个
        resized_image = scipy.misc.imresize(
            resized_image,
            max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(  # preprocess_image 是一个对图像进行处理的函数，看
        resized_image,
        preprocessing="torch"
    )
    with torch.no_grad(): # 不使用 with torch.no_grad() 此时grad_fn 有属性，计算的结果在一计算图当中，可以进行梯度反转等操作，当使用这个语句，表示不需要反向传播，但其实两个结果都是一样的
        if multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    if nfeatures != -1:
        #根据scores排序
        scores2 = np.array([scores]).T
        res = np.hstack((scores2, keypoints))
        res = res[np.lexsort(-res[:, ::-1].T)]

        res = np.hstack((res, descriptors))
        #取前几个
        scores = res[0:nfeatures, 0].copy()
        keypoints = res[0:nfeatures, 1:4].copy()
        descriptors = res[0:nfeatures, 4:].copy()
        del res
    return keypoints, scores, descriptors
