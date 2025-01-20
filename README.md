# Project Overview

This repository combines two projects to offer a solution for detecting suspicious activity through pose estimation and behavioral analysis.

- [HouraHashemi/mmpose_xgb](https://github.com/HouraHashemi/mmpose_xgb)
- [Suspicious Activity Detection Using YOLOv8 Pose Estimation](https://medium.com/@sg.sparsh06/suspicious-activity-shoplifting-detection-using-yolov8-pose-estimation-and-classification-b59fd73cdba3)

## Steps to Use

1. **Extract Frames from Video**  
   Use the `2d_to_3d.ipynb` script to extract individual frames from the video.

2. **Organize the Data**  
   Manually categorize the extracted frames into two folders:  
```
  .
  └── dataset
      ├── Normal
      └── Suspicious
```
3. **Train the XGBoost Model**  
Extract 3D pose data (angles) from the images and use them to train the XGBoost model.

4. **Behavioral Detection**  
Once the XGBoost model is trained, it is ready to detect and classify behaviors as either normal or suspicious.
