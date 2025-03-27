# Hybrid integrated feature fusion of handcrafted and deep features for rice blast resistance identification using UAV imagery
Pytorch implementation of Hybrid integrated feature fusion of handcrafted and deep features for rice blast resistance identification using UAV imagery.
Rice Blast Disease Detection System for Rice Seedlings Based on Hybrid Image Feature Fusion, combining EfficientNet and texture features for multimodal classification.


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

Data and the pretrained weights of the model can be accessed via the following methods:

Link of the data: https://anxia.com/s/swh5xl73wqv?password=cca6&#
data.zip

Access Code: cca6

Link of the pretrained weights: https://anxia.com/s/swh5xlq3wqv?password=d456&#
HIFF.pth

Access Code: d456

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


