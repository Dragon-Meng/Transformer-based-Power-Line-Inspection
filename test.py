import warnings
from ultralytics import RTDETR

# Ignore warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load the trained model
    model = RTDETR('/path/to/your/model/weights/best.pt')  # Replace with your own model weight path

    # Run validation
    model.val(
        data='/path/to/your/dataset/data.yaml',  # Path to the dataset configuration file
        split='test',                           # Choose dataset subset: train, val, or test
        imgsz=640,                              # Input image size
        batch=16,                               # Batch size
        # save_json=True,                       # Uncomment to calculate COCO metrics if needed
        project='output',                       # Output directory
        name='experiment_name',                 # Experiment name (use a meaningful name here)
    )
