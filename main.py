import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

def plotfig(img, x_main, y_main, w, h, color):
    plt.figure(figsize=(9,9))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for bb in range(len(x_main)):
        rectangle = patches.Rectangle((x_main[bb],y_main[bb]), w[bb], h[bb], fill=False, edgecolor=color)
        plt.gca().add_patch(rectangle)
    plt.axis("off")
    plt.title(f'Scene ID: {scenes:06d}')
    plt.show()

root = "ComputerVision/Object-Detection-and-Depth-Estimation/KITTI_Selection"

scene_list = [6037,6042,6048,6054,6059,6067,6097,6098,6121,6130,6206,6211,6227,6253,6291,6310,6312,6315,6329,6374]

for scenes in scene_list:
    img = cv2.imread(f"{root}/images/{scenes:06d}.png")
    calib = np.loadtxt(f"{root}/calib/{scenes:06d}.txt")
    labels = np.loadtxt(f"{root}/labels/{scenes:06d}.txt", usecols=(1,2,3,4,5))
    # Load a model
    model = YOLO('yolov8x.pt')  # load an official model
    #model = YOLO('path/to/best.pt')  # load a custom model
    
    # Predict with the model
    results = model(img, classes= [2])  # predict on an image
    bb_result = results[0].boxes.xywh.cpu().numpy().astype(int)

    x_main = bb_result[:,0]-bb_result[:,2]/2
    y_main = bb_result[:,1]-bb_result[:,3]/2
    w = bb_result[:,2]
    h = bb_result[:,3]
    
    plotfig(img, x_main, y_main, w, h, 'red')
    
    x_dist = bb_result[:,0]
    y_dist = bb_result[:,1]+bb_result[:,3]/2
    
    obj_image = np.array([[x_dist[i], y_dist[i], 1] for i in range(len(x_dist))])
    pred_dist = np.array([])
    
    for i in range(len(x_dist)):
        obj_image[i,:] = np.dot(np.linalg.inv(calib), obj_image[i,:])  # Apply intrinsic camera matrix
        c = 1.65/obj_image[i,:][1]
        
        obj_image[i,:][0] = obj_image[i,:][0]*c
        obj_image[i,:][1] = obj_image[i,:][1]*c
        obj_image[i,:][2] = obj_image[i,:][2]*c
        
        pred_dist = np.append(pred_dist, np.sqrt(np.sum(np.square((obj_image[i,:][0],obj_image[i,:][1],obj_image[i,:][2])))))
        
    print(obj_image)
    print(pred_dist)
    
    plotfig(img, labels[:,0],labels[:,1],(labels[:,2]-labels[:,0]),(labels[:,3]-labels[:,1]), 'green')
