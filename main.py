import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

root = "KITTI_Selection"

scene_list = [6037,6042,6048,6054,6059,6067,6097,6098,6121,6130,6206,6211,6227,6253,6291,6310,6312,6315,6329,6374]

for scenes in scene_list:
    img = cv2.imread(f"{root}/images/{scenes:06d}.png")
    
    plt.figure(figsize=(9,9))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f'Scene ID: {scenes:06d}')
    plt.show()