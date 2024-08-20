# gpu version for training
# -1 means: disable cuda/gpu only using cpu
# otherwise the number will indicate gpu id
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# libary 
import pandas as pd
import cv2
import shutil
import yaml
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch
from IPython.display import display, Image
import sys

#  model using yolov8
class YOLOv8:
    def __init__(self):

        # Load model
        self.model = YOLO('yolov8s.pt')
        # load path

        self.path_to_data = "./data/ts/ts"


        # size of image
        self.width = 676
        self.height = 380
        # txt path
        self.train_data_txt = './data/train.txt'
        self.test_data_txt = './data/test.txt'
        self.train_label_txt = './data/train_label.txt'
        self.test_label_txt = './data/test_label.txt'


        # directory
        self.root_dir = './datasets'
        self.labels_dir = './labels'
        self.images_dir = './images'

        # # Create directories if they don't exist
        # os.makedirs(self.labels_dir + './datasets/train', exist_ok=True)
        # os.makedirs(self.labels_dir + './datasets/val', exist_ok=True)
        # os.makedirs(self.images_dir + './datasets/train', exist_ok=True)
        # os.makedirs(self.images_dir + './datasets/val', exist_ok=True)

        # path to video prediction
        self.video_pred = './data/traffic-sign-to-test.mp4'

    def _load_data(self):
        # load image
        with open(self.train_data_txt, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                shutil.copy(line.strip(), self.images_dir + '/train')
        with open(self.test_data_txt, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                shutil.copy(line.strip(), self.images_dir + '/val')

        # load label
        with open(self.train_label_txt, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                shutil.copy(line.strip(), self.labels_dir + '/train')
        with open(self.test_label_txt, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                shutil.copy(line.strip(), self.labels_dir + '/val')
        # yolo format data 
        # name of class prohibitory, danger, mandatory, other
        yolo_format_data = {
            'path': self.root_dir,
            'train': "./datasets/images/train",
            'val': "./datasets/images/val",
            'nc': 4,
            'names': ['prohibitory', 'danger', 'mandatory', 'other']
        }
        with open('./datasets/yolo.yaml', 'w') as outfile:
            yaml.dump(yolo_format_data, outfile, default_flow_style=False)
        print("Data loaded successfully!")
    

    # Train the model method
    def train_model(self, epochs, batch_size):
        self.model.train(
            data='./datasets/yolo.yaml',
            patience = 5,
            epochs=epochs,
            batch=batch_size,
            workers=3
        )
    def val(self):
        path_best_weights="./runs/detect/train34/weights/best.pt"
        model = YOLO(path_best_weights) 
        metrics = model.val() 
        print(f'mean average precision @ .50: {metrics.box.map50}')

    def predict(self):
        prediction_dir = './datasets/predictions'
        path ="./runs/detect/train34/weights/best.pt"
        model = YOLO(path)
        result = model.predict(source = self.video_pred , conf=0.5, iou=0.75, save=True, save_txt=True)

    def show_result(self):
        path_result = './runs/detect/predict/traffic-sign-to-test.avi'
        display(Image(path_result))


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        yolo = YOLOv8()
        yolo._load_data()
        yolo.train_model(epochs=10, batch_size=15)
    elif sys.argv[1] == 'val':
        yolo = YOLOv8()
        yolo.val()
    elif sys.argv[1] == 'predict':
        yolo = YOLOv8()
        yolo.predict()
    elif sys.argv[1] == 'show_result':
        yolo = YOLOv8()
        yolo.show_result()



# D:/Python/sup/YOLO/data/testing_images/vid_5_26640.jpg

# D:/Python/sup/YOLO/datasets/predictions/vid_5_26640.txt