<br>
# Model used
<br>
<br>
<p>
YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

</div>
This project is based on the Pytorch implementation of YOLOv5.

`detect.py` runs YOLOv5 inference on a variety of sources. `yolov5_soccer_player_tracking.ipynb` downloads models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases), and saves results to `runs/detect`.

# High level overview of how it works:
- The pretrained weights of the Yolov5 model work quite neat.
- We incorporate SORT (Simple Online and Realtime Tracking) [SORT](https://github.com/abewley/sort) algorithm for object tracking. 
- Transform the corner field view to a 2D top view of the field.


# Conclusion:
Yolov5 is quite accurate in tracking the ball and the players.

# Uses of such tracking:
1. Analyzing gameplay paths for any player
2. Finding average speed of any player