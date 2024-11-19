import sys
sys.path.append("ByteTrack")

from yolox.tracker.byte_tracker import BYTETracker, STrack
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from typing import List, Optional

def get_video_frames(video_path):

    video = cv2.VideoCapture(str(video_path))

    frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)

    video.release()

    return frames

video_path = Path("/home/urvashi2022/Desktop/UI_DEVELOPMENT/inputv.mp4")

frames = get_video_frames(video_path)

path_weights = "/home/urvashi2022/Desktop/UI_DEVELOPMENT/tracking/best300.pt"
def load_yolo_model(path_weights):
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path_weights, device="cpu", force_reload=True)
    return yolo_model

yolo_model = load_yolo_model(path_weights)

@dataclass
class Detection:
    xywh: List[float]
    xyxy: List[float]
    class_id: int
    class_name: str
    confidence: float
    tracker_id: Optional[int] = None

    @classmethod
    def from_results(cls, pred):
        
        result = []
                
        ind_to_cls = {
            0: "ball",
            1 : "goalkeeper",
            2 : "player",
            3 : "referee"
        }

        for x_min, y_min, x_max, y_max, confidence, class_id in pred:
            class_id=int(class_id)
            result.append(Detection(
                xywh=[float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                xyxy=[float(x_min), float(y_min), float(x_max), float(y_max)],
                class_id=class_id,
                class_name=ind_to_cls[class_id],
                confidence=float(confidence)
            ))
        return result
    
def draw_detections(image, detections, draw_tacker_id: bool = False):
    image = image.copy()
  
    colors = {
        "ball": (0,200,200), # yellow
        "player": (255,0,0), # blue
        "goalkeeper":(255,0,255), # magenta
        "referee": (0,0,255) # red
    }

    colors1 = {
        "Team1" : (0, 0, 0),
        "Team2" : (255, 255, 255)
        }
    for pred in detections:
        bbox = pred.xyxy
        cls = pred.class_name
        #cv2.rectangle(img=image, pt1=tuple([int(b) for b in bbox[:2]]), pt2=tuple([int(b) for b in bbox[2:]]), color=colors[cls], thickness=3)
        
        center_bottom = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
        bbox_width = int(bbox[2] - bbox[0])
        
        cv2.ellipse(image, center_bottom, (bbox_width // 2, 8), 0, 0, 180, color=colors[cls], thickness=3)
            
        if draw_tacker_id and cls != "ball":
            cv2.putText(image, str(pred.tracker_id), (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[cls], 3)
        else:
            cv2.putText(image, cls, (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[cls], 3)

    return image

#prediction = yolo_model(frames[0]).pred[0].cpu().numpy()

#image = draw_detections(frames[0], Detection.from_results(prediction))

from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
  track_thresh: float = 0.25
  track_buffer: int = 30
  match_thresh: float = 0.8
  aspect_ratio_thresh: float = 3.0
  min_box_area: float = 1.0
  mot20: bool = False

#byte_tracker = BYTETracker(BYTETrackerArgs)

def format_predictions(predictions, with_conf: bool = True):
  """
  Format yolo detection to ByteTracke format: (x1, y1, x2, y2, conf)
  """
  frame_detections = []
  for pred in predictions:
      bbox = pred.xyxy
      conf = pred.confidence
      if with_conf:
        detection = bbox + [conf]
      else:
        detection = bbox

      frame_detections.append(detection)
  return np.array(frame_detections, dtype=float)

def match_detections_with_tracks(detections, tracks):
  """
  Find which tracker corresponds to yolo detections and set the tracker_id.
  We compute the iou between the detection and trackers.
  """
  detections_bboxes = format_predictions(detections, with_conf=False)
  tracks_bboxes = np.array([track.tlbr for track in tracks], dtype=float)
  iou = box_iou_batch(tracks_bboxes, detections_bboxes)
  track2detection = np.argmax(iou, axis=1)

  for tracker_index, detection_index in enumerate(track2detection):
    if iou[tracker_index, detection_index] != 0:
      detections[detection_index].tracker_id = tracks[tracker_index].track_id
  return detections

def get_video_writer(output_video_path, fps, width, height):
  """
  Create a video writer to save new frames after annotation
  """
  output_video_path.parent.mkdir(exist_ok=True)
  return cv2.VideoWriter(
      str(output_video_path),
      fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
      fps=fps,
      frameSize=(width, height),
      isColor=True
  )

video_path = Path("/home/urvashi2022/Desktop/UI_DEVELOPMENT/inputv.mp4")
output_video_path = Path("/home/urvashi2022/Desktop/UI_DEVELOPMENT/tracking/output1.mp4")

def get_output_video(input_video_path, output_video_path):
   
    byte_tracker = BYTETracker(BYTETrackerArgs)

    video_writer = get_video_writer(
        output_video_path,
        30,
        frames[0].shape[1],
        frames[0].shape[0]
    )

    for frame in tqdm(frames[:400]):

        # detect players with yolo
        detections = yolo_model(frame).pred[0].cpu().numpy()

        detections = Detection.from_results(detections)

        # create a new list of detection with tracker_id attribute.
        detections_with_tracker = []
        for detection in detections:
            detection.tracker_id = ""
            detections_with_tracker.append(detection)

        # get trackers with ByteTrack
        tracks = byte_tracker.update(
            output_results=format_predictions(detections_with_tracker, with_conf=True),
            img_info=frame.shape,
            img_size=frame.shape
        )

        # set tracker_id in yolo detections
        detections_with_tracker = match_detections_with_tracks(detections_with_tracker, tracks)

        # annotate the frame
        image = draw_detections(frame, detections_with_tracker, True)

        # save the frame to video writer
        video_writer.write(image)

    # save the video
    video_writer.release()

final = get_output_video(video_path, output_video_path)