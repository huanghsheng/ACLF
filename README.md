# Adaptive connection line filtering for image registration of UAV-based visual and thermal infrared imagery in rice phenotyping
Pytorch realizes the feature extraction of visible and thermal infrared images, which is used to realize the registration of visible and thermal infrared images of UAV images. 
Based on the dual mode registration system of visible light and thermal infrared, an adaptive filtering algorithm is used for registration.


## Project Structure


├── config.py        # Configuration parameters

├── dataset.py       # Dataset loading and processing

├── model.py         # HIFF model definition

├── train.py         # Training script

└── utils.py         # Utility functions


## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV (cv2)
- mahotas
- pandas
- scikit-learn
- CUDA (recommended for training)

## Data Preparation

The dataset should be organized as follows:


data/

├── 606_pro/            # Original images

│   ├── 0.png

│   ├── 1.png

│   └── ...

├── 606_pro_CLAHE_1/    # CLAHE enhanced images

│   ├── 0.png

│   ├── 1.png

│   └── ...

├── train_3_pro.xlsx    # Training labels file

└── texture_features.csv # Texture features file (auto-generated)



## Key Features

- Multimodal feature fusion (image features + texture features)
- EfficientNet-B0 based deep learning model
- CLAHE image enhancement
- Automatic texture feature extraction (IDM, entropy, contrast)
- Weighted loss function support
- Adaptive learning rate adjustment

## Usage Guide

Data of the model can be accessed via the following methods:

Link of the data: https://115.com/?cid=3120429764003117178&offset=0&tab=&mode=wangpan&#
section.zip


### 1. Configuration

Set key parameters in `config.py`:

```python
BATCH_SIZE = 18
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
STEP_SIZE = 8
GAMMA = 0.5
CLASS_WEIGHTS = [2, 4, 6]
TRAIN_TEST_SPLIT = [0.7, 0.3]

```

### 2. Data Processing
The system automatically processes:

Original and CLAHE enhanced images
Haralick texture feature extraction
Feature CSV file generation


### 3. Model Training
Run the training script:
```python
python train.py
```

Training outputs:

Loss value per epoch
Accuracy
Recall
F1 score

Model saving:

Best model saved as best_train_model.pth
Final model saved as final_model.pth

### 4. Moedel Testing
Run the testing script:
```python
python test.py
```

### 5. Model Architecture
HIFF  model includes:

EfficientNet-B0 backbone
Dual-path feature fusion layer
Multi-layer fully connected classifier
HIFF Structure:
```markdown
EfficientNet-B0 --> FC(1000->64) --→  

                                      Concat --> FC(128->3)  
                                      
Texture Features --> FC(3->64)   --→

```

Citation
If you use this code in your research, please cite:
@article{
  title={Hybrid integrated feature fusion of handcrafted and deep features for rice blast resistance identification using UAV imagery},
  author={Peng Zhang, Zibin Zhou, Huasheng Huang, Yuanzhu Yang, Xiaochun Hu, Jiajun Zhuang, and Yu Tang},
}
License
MIT License

Contact
For any questions, please reach out through:

Submit an Issue
Email: huanghsheng@gpnu.edu.cn


