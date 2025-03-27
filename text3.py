import scipy.io as sio

input_file='./shougong_baiyun_mat1/143_1.json'
# output_file='./xiaorong/7/137_3.json'
output_file='1.json'
# output_T='./xiaorong/7/137_T3.mat'
# output_W='./xiaorong/7/137_W3.mat'

output_T='ACLF_T1.mat'
output_W='ACLF_W1.mat'

mat1 = sio.loadmat('matrix1.mat')
mat2= sio.loadmat('matrix3.mat')


random_rows1 = mat1['matrix']
random_rows2 = mat2['P3']
print(random_rows1)
print(random_rows2)



import cv2 as cv

ransacReprojThreshold = 4
H, status =cv.findHomography(random_rows1, random_rows2, cv.RANSAC,ransacReprojThreshold)
print(H)


import numpy as np
import json

with open(input_file, 'r') as file:
    data = json.load(file)

random_rows3 = data['ir_points']
random_rows5 = data['rgb_points']

i=0
x=[]
y=[]
import numpy as np
random_rows3=np.array(random_rows3)
for i in range(random_rows3.shape[0]):
    a=random_rows3[i, 0]
    b=random_rows3[i, 1]
    point_in = np.array([[a, b, 1]])
    point_out = np.dot(H, point_in.T)
    x_out, y_out, _ = np.array(point_out).reshape(-1) / point_out[2]
    x.append(x_out)
    y.append(y_out)
random_rows4=np.c_[x,y]
print(f"random_rows4: ({random_rows4})")

random_rows5=np.array(random_rows5)
dist=np.sum(np.sqrt(np.sum((random_rows4 - random_rows5) ** 2, axis=1)),axis=0)/random_rows4.shape[0]
print('dist:')
print(dist)

import json

data = {
            "homography_matrix": H.tolist(),
            "error": dist
        }
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)
print(f"Data saved to {output_file}")

# save random_rows4 and random_rows5  is .mat,in order to calculate dist
from scipy.io import savemat
savemat(output_T, {'matrix': random_rows4})
savemat(output_W, {'matrix': random_rows5})





