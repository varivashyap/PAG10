# Ultralytics YOLOv8

## Overview

YOLOv8 is the latest iteration in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. Building upon the advancements of previous YOLO versions, YOLOv8 introduces new features and optimizations that make it an ideal choice for various [object detection](https://www.ultralytics.com/glossary/object-detection) tasks in a wide range of applications.

![Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/yolov8-comparison-plots.avif)

# Event Detection Using YOLOv8

## YOLOv8 training

The first part of the project is to train YOLOv8 on detecting events in images. The model was trained for 10 epochs on 204 images with a resolution of 800x800.

The classes the model is trained to predict are Corner-kick, Free-kick, Red-card, Shooting, Substitution, Yellow-card.

In the second part, the code extracts frames from input video and the trained model makes its prediction on each frame. Then, a .csv file is created to store the counts of each event.
