# Adaptive connection line filtering for image registration of UAV-based visual and thermal infrared imagery in rice phenotyping
Pytorch realizes the feature extraction of visible and thermal infrared images, which is used to realize the registration of visible and thermal infrared images of UAV images. 
Based on the dual mode registration system of visible light and thermal infrared, an adaptive filtering algorithm is used for registration.


## Project Structure


├── cnnmatching.py      

├── cp_subpixelFine.m      

├── plotmatch.py      

├── show_resultimage.m       

├── test3.m     

└── text3.py  


## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV (cv2)
- mahotas
- pandas
- scikit-learn
- CUDA (recommended for training)
- matlab

## Data Preparation

The dataset should be organized as follows:

baiyun/       # Original images

├── 137_W1

├── 137_W2

├── 137_W3

├── 137_T1

└── ...

baiyun_gray_size/       # Gray, uniform size

├── 137_W1

├── 137_W2

├── 137_W3

├── 137_T1

└── ...


baiyun_gray_size_dbd/       #  Gray, uniform size， Contrast enhancement

├── 137_W1

├── 137_W2

├── 137_W3

├── 137_T1

└── ...

## Key Features
-Multimodal image registration (visible image+thermal infrared image) 

-Algorithm based on adaptive filtering

## Usage Guide

Data of the model can be accessed via the following methods:

Link of the data: https://115.com/?cid=3120429764003117178&offset=0&tab=&mode=wangpan&#
section.zip


### 1. Configuration

Set key parameters in `config.py`:

```python
BATCH_SIZE = 4
c = 0.33
d = 0.33

```

### 2. Data Processing
The system automatically processes:

-Panorama stitching  

-High precision cell clipping 

-Contrast enhancement, uniform size, etc



### 3. Model Architecture
ACLF  model includes:

-Contrast enhancement 

-Adaptive connector filtering 

-Fine grained matching

Citation

If you use this code in your research, please cite:

@article{

  title={Adaptive connection line filtering for image registration of UAV-based visual and thermal infrared imagery in rice phenotyping},
  
  author={Yu Tang, Jiajun Xu, Huasheng Huang, Jiajun Zhuang, Zibin Zhou, Peng Zhang},
  
}

-License

-MIT License

Contact

For any questions, please reach out through:

Submit an Issue

Email: huanghsheng@gpnu.edu.cn


