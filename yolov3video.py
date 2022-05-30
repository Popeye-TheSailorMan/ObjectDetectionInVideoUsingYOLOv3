# What we are going to do
# Reading input video -> Loading YOLOv3 -> reading
# frame in the loop -> getting blob of the frame ->
# implementing forward pass -> getting bounding boxes ->
# non- maximum suppression -> drawing bounding boxes ->
# writing processed frames

import cv2
import numpy as np
import time

#Read video file
vid = cv2.VideoCapture('videos/traffic-cars.mp4')

#importing yolo
with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

yolo = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                  'yolo-coco-data/yolov3.weights')
all_layers = yolo.getLayerNames()

#get output layers
output_layers = \
    [all_layers[i-1] for i in yolo.getUnconnectedOutLayers()]
minimum_probability = 0.5
threshold = 0.3

colours = np.random.randint(0,255,size=(len(labels),3),dtype='uint8')

f = 0
t = 0
h,w = None, None

#Getting frame one by one
#Implementing Blob
while True:
    ret, frame = vid.read()

    if not ret:
        break
    if w is None or h is None:
        h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop = False)

#Implementing Forward pass

    yolo.setInput(blob)
    start = time.time()
    output = yolo.forward(output_layers)
    end = time.time()
    print(end-start)

    f += 1
    t += end - start

#Getting bounding boxes

    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current>probability_minimum:
                box_current = detected_objects[0:4]*np.array([w,h,w,h])
                x_center, y_center_box_width,box_width = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
