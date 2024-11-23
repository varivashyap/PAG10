# PERFORMANCE ANALYZER

Goal: To analyze a soccer match by tracking the players, the ball, and the referee and extracting 
useful performance metrics that help assess the game 

Target user: Coaches and Players 

Input: A video of a soccer match 

Output:  
- Video identifying and tracking important elements on the field 
- 2-D Bird’s Eye View  
- Performance Metrics and Analysis
  
Datasets:
SoccerNet
Roboflow Soccer Match Data

## Multiple Object Detection and Tracking 
1. Multiple Object Detection (YOLOv5):
Detecting significant objects such as players from the input video  
2. Team Affiliation:
Team affiliation (K-Means clustering algorithm)
3. Tracking (ByteTrack):
Create tracklets of players by associating the detections across frames to track players 
and assign unique ids to them 
4. Tracklet Consistency:  
Ensures consistency over time for various attributes using majority voting to correct any 
momentary detection errors

## Bird’s Eye View 
1. Pitch Localization: Localizing pitch by detecting the field's lines 
2. Camera Calibration: Calibrating the camera for translating image coordinates 
into real-world coordinates 
3. Perspective Transformation: Transforming tracklets into 2D positions on the pitch based 
on a homography matrix

## Performance Metrics 
The following metrics are to be extracted from the video. 
- Speed (calculated using the Bird’s Eye View with player identification and tracking) 
- Pass, Header, High pass, Out, Shot, Interception, Goal (using labelled data from the 
SoccerNet Ball-Action Spotting Dataset) (ResNet-512) 
- Shots on target, Shots off target, Fouls, Corner, Yellow card, Red card, Yellow to red card 
(using labelled data from the SoccerNet Action Spotting Dataset) (ResNet-512) 
- Possession (Based on pass and interception data)  
Performance Metrics Analysis 
We used a Gemini 1.5 Flash API Key to report some basic statistical analysis of the metrics data 
that was extracted previously. 
User Interface 
We used Streamlit to create the website.
