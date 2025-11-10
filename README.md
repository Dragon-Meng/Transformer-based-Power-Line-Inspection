# Transformer-based-Power-Line-Inspection

This project implements a Transformer-based model for efficient UAV-based power line inspection, specifically designed to handle multi-scale objects. Paper link: 【】was published in：【】

### 1. Virtual Environment Setup and Git Clone the Project

First, ensure that you have **CUDA** and **PyTorch** installed, and set up an appropriate virtual environment. Then, use the following commands to clone the project:

```bash
git clone https://github.com/yourusername/Transformer-based-Power-Line-Inspection.git
cd Transformer-based-Power-Line-Inspection
```

### 2. Dataset Download

You need to download two datasets:

1. **InsPLAD Dataset**: This dataset contains power line images and corresponding annotations, suitable for multi-scale object detection tasks. Download link: [InsPLAD Dataset](https://github.com/andreluizbvs/InsPLAD)

2. **PTL-AI_Furnas Dataset**: This dataset contains image data from power lines, suitable for power line inspection tasks. Download link: [PTL-AI_Furnas Dataset](https://github.com/freds0/PTL-AI_Furnas_Dataset)

After downloading, place the datasets in the `dataset/` folder, ensuring the directory structure is as follows:

```plaintext
dataset/
├── InsPLAD/
│   ├── images/
│   └── labels/
└── PTL-AI_Furnas/
    ├── images/
    └── labels/
```

Configure `dataset/data.yaml` with the correct paths:

```yaml
train: ./dataset/InsPLAD/images/train  # Training set path
val: ./dataset/InsPLAD/images/val      # Validation set path
test: ./dataset/InsPLAD/images/test    # Test set path
```

### 3. Data Format and Splitting

1. **Data Format Conversion**: Convert the data to **YOLO format** (compatible with Ultralytics' official YOLO version). Ensure that the image and label files are saved in YOLO format.

2. **Dataset Splitting**: Split the dataset randomly into training, validation, and test sets according to the dataset size. For example, 70% for training, 15% for validation, and 15% for testing.

### 4. Environment Setup

Follow these steps to set up your environment:

1. **Install Dependencies**: The required dependencies are listed in the `requirements.txt` file. Install them using the following command:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Ultralytics**: Download and install **Ultralytics**, which provides the baseline model **RT-DETRv2** along with its code and weights:

   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics
   pip install -r requirements.txt
   ```

   You can use Ultralytics' pretrained models or train your own model using the prepared datasets. **All models in this project are trained from scratch, and it is recommended not to use pretrained weights**.

---

### 5. Modify the Model

In the `Transformer-based-Power-Line-Inspection/models` directory, locate and modify the relevant Python files, integrating them with the Ultralytics module to adapt to the RT-DETRv2 structure. You can refer to the following approach:

For example, add the `BasicBlock_APFA` module in `ultralytics\nn\modules\block.py`, and make similar changes to other files. Ensure that these improvements are compatible with the existing RT-DETRv2 structure.

### 6. Configure Your Config File and Start Training

Once the model modifications are complete, configure your training config file and then start training and testing with the following commands:

#### **Training Command**:

```bash
yolo train model=none data=dataset/data.yaml epochs=150 imgsz=640 batch=16 cfg=config/your_config.yaml

```

#### **Testing Command**:

```bash
yolo test model=./checkpoints/ourextramodel.pth data=dataset/data.yaml imgsz=640 batch=16
```

---

### 7. Acknowledgments

We would like to thank all the developers who contributed to the open-source community, especially the **Ultralytics** team for providing the RT-DETRv2 model, which greatly accelerated the development of this project.

### 8. Citation

Please cite this project as follows:

```

```

---

