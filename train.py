import warnings
import argparse
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for RT-DETRv2 model.")

    parser.add_argument('--data', type=str, required=True, help="./dataset/data.yaml path")
    parser.add_argument('--epochs', type=int, default=150, help="Number of epochs for training")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Input image size")
    parser.add_argument('--project', type=str, default='project', help="Project name")
    parser.add_argument('--name', type=str, default='FUxian', help="Experiment name")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

  
    model = RTDETR('./config/your_model_config.yaml')

    
    results = model.train(
        data=args.data,               
        epochs=args.epochs,           
        batch=args.batch,             
        imgsz=args.imgsz,             
        save=True,                    
        project=args.project,         
        name=args.name,               
        pretrained=False,            
        optimizer='AdamW',           
        val=True,                     
    )
