# Event Detection Using YOLOV5

## YOLOv8 training

The first part of the project is to train YOLOv8 on detecting events in images. The model was trained for 10 epochs on 204 images with a resolution of 800x800.

The classes the model is trained to predict are Corner-kick, Free-kick, Red-card, Shooting, Substitution, Yellow-card.

In the second part, the code extracts frames from input video and the trained model makes its prediction on each frame. Then, a .csv file is created to store the counts of each event.
