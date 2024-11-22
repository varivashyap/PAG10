# Ultralytics YOLOv8

## Overview

YOLOv8 is the latest iteration in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. Building upon the advancements of previous YOLO versions, YOLOv8 introduces new features and optimizations that make it an ideal choice for various [object detection](https://www.ultralytics.com/glossary/object-detection) tasks in a wide range of applications.

## Key Features

- **Advanced Backbone and Neck Architectures:** YOLOv8 employs state-of-the-art backbone and neck architectures, resulting in improved [feature extraction](https://www.ultralytics.com/glossary/feature-extraction) and object detection performance.
- **Anchor-free Split Ultralytics Head:** YOLOv8 adopts an anchor-free split Ultralytics head, which contributes to better accuracy and a more efficient detection process compared to anchor-based approaches.
- **Optimized Accuracy-Speed Tradeoff:** With a focus on maintaining an optimal balance between accuracy and speed, YOLOv8 is suitable for real-time object detection tasks in diverse application areas.
- **Variety of Pre-trained Models:** YOLOv8 offers a range of pre-trained models to cater to various tasks and performance requirements, making it easier to find the right model for your specific use case.


# Event Detection Using YOLOv8

## YOLOv8 training

The first part of the project is to train YOLOv8 on detecting events in images. The model was trained for 10 epochs on 204 images with a resolution of 800x800.

The classes the model is trained to predict are Corner-kick, Free-kick, Red-card, Shooting, Substitution, Yellow-card.

In the second part, the code extracts frames from input video and the trained model makes its prediction on each frame. Then, a .csv file is created to store the counts of each event.
