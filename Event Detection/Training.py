


from IPython import display
display.clear_output()

import ultralytics

ultralytics.checks()
from ultralytics import YOLO

from IPython.display import display, Image
import torch

dataLoc = '/home/urvashi2022/Desktop/UI_DEVELOPMENT/DetectionMoreEvents/dataset/data.yaml'


model = YOLO("yolov8n.pt")

epochs = 10
size = 800
batch = 16
task = "detect"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = False
mode = "train"
model.train(data=dataLoc, epochs=epochs, imgsz=size, device=device, batch=batch, task=task, plots=True, mode=mode, amp=amp)


Image(filename='/home/urvashi2022/Desktop/UI_DEVELOPMENT/DetectionMoreEvents/runs/detect/train5/confusion_matrix.png')


Image(filename='/home/urvashi2022/Desktop/UI_DEVELOPMENT/DetectionMoreEvents/runs/detect/train5/results.png', width=600)


Image(filename='/home/urvashi2022/Desktop/UI_DEVELOPMENT/DetectionMoreEvents/runs/detect/train5/val_batch0_pred.jpg', width=600)

