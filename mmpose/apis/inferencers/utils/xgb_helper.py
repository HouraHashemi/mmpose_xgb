import csv
import torch
import numpy as np
import hashlib
import os
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
import numpy as np
import pandas as pd
from itertools import combinations
import pandas as pd
import os
import xgboost as xgb

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import cv2
from ultralytics import YOLO
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_frames_keypoints_as_list(all_frames_results):
    return [frame_result['predictions'][0][0]['keypoints'] for frame_result in all_frames_results]


def store_3d_data(frames, file_path):

    # Open the file for writing
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header (column names)
        # Assuming frames are 3D, with the format (num_frames, num_keypoints, 3)
        num_keypoints = frames.shape[1]  # Number of keypoints
        header = ['image_name']  # Add 'image_name' as the first column header
        for i in range(num_keypoints):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])  # Group x, y, z for each keypoint
        writer.writerow(header)

        # Write each frame's data
        for frame_id, frame in enumerate(frames):
            # Construct the image name for the current frame
            image_name = f'person_nn_{frame_id}.jpg'  # Format frame number with leading zeros (e.g., frame_000, frame_001)
            
            # Flatten the 3D pose (each keypoint has 3 values: x, y, z)
            row = [image_name]  # Add the frame name as the first element of the row
            for keypoint in frame:
                row.extend(keypoint.flatten())  # Add x, y, z for each keypoint
            writer.writerow(row)




def calculate_angle(a, b, c):     
        # Vectors BA and BC
    v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])  # BA
    v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])  # BC

    # Calculate the dot product
    dot_product = np.dot(v1, v2)

    # Calculate magnitudes
    v1_magnitude = np.linalg.norm(v1)
    v2_magnitude = np.linalg.norm(v2)

    # Cosine of the angle
    cos_theta = dot_product / (v1_magnitude * v2_magnitude)

    # Numerical stability: ensure cos_theta is in range [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate angle in radians and convert to degrees
    angle_rad = np.arccos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def calculate_and_store_angles(frames_points, csv_filename):
    # Prepare to write to the CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header row if the file is empty
        if file.tell() == 0:
            # Generate the column names for the angles
            angle_columns = []
            num_points = 17
            for (i, j, k) in combinations(range(num_points), 3):
                angle_columns.append(f"{i+1}-{j+1}-{k+1}")
            
            # Write the header row with file name and angle columns
            writer.writerow(['file_name'] + angle_columns)

        # Loop over each frame and calculate angles
        for frame_idx, points in enumerate(frames_points, start=1):
            print(frame_idx)
            frame_id = f"person_nn_{frame_idx}.jpg"  # Frame name like 'person_nn_1', 'person_nn_2', etc.

            # List to store angles for this frame
            angles = []

            # Generate all combinations of three points (17 points -> 680 combinations)
            for (i, j, k) in combinations(range(len(points)), 3):
                point1 = points[i]
                point2 = points[j]
                point3 = points[k]

                # Calculate the angle between points i, j, k (with point j as the middle point)
                angle = calculate_angle(point1, point2, point3)
                angles.append(angle)

            # Write the frame_id and corresponding angles to the CSV
            writer.writerow([frame_id] + angles)

    print(f"CSV file '{csv_filename}' has been updated successfully.")



def calculate_and_return_angles(frames_points):
    # Initialize a list to store all the data
    data = []

    # Generate the column names for the angles
    num_points = 17
    angle_columns = [f"{i+1}-{j+1}-{k+1}" for i, j, k in combinations(range(num_points), 3)]

    # Loop over each frame and calculate angles
    for frame_idx, points in enumerate(frames_points, start=1):
        print(frame_idx)
        frame_id = f"person_nn_{frame_idx}.jpg"  # Frame name like 'person_nn_1', 'person_nn_2', etc.
        
        # List to store angles for this frame
        angles = []

        # Generate all combinations of three points (17 points -> 680 combinations)
        for i, j, k in combinations(range(len(points)), 3):
            point1 = points[i]
            point2 = points[j]
            point3 = points[k]

            # Calculate the angle between points i, j, k (with point j as the middle point)
            angle = calculate_angle(point1, point2, point3)
            angles.append(angle)

        # Append the frame_id and corresponding angles to the data list
        data.append([frame_id] + angles)

    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=['file_name'] + angle_columns)

    return df



def calculate_one_frame_angles(frames_points):

    # List to store angles for this frame
    angles = []

    # Generate all combinations of three points (17 points -> 680 combinations)
    for (i, j, k) in combinations(range(len(frames_points)), 3):
        point1 = frames_points[i]
        point2 = frames_points[j]
        point3 = frames_points[k]

        # Calculate the angle between points i, j, k (with point j as the middle point)
        angle = calculate_angle(point1, point2, point3)
        angles.append(angle)
    
    return angles



def angles_keys():
    
    angle_columns = []
    num_points = 17
    for (i, j, k) in combinations(range(num_points), 3):
        angle_columns.append(f"{i+1}-{j+1}-{k+1}")

    return angle_columns
        


def labeling_dataframe(df, dataset_path):

    sus_path = os.path.join(dataset_path, 'Suspicious')
    normal_path = os.path.join(dataset_path, 'Normal')


    def get_label(image_name, sus_path, normal_path):

        if image_name in os.listdir(sus_path):
            return 'Suspicious'
        elif image_name in os.listdir(normal_path):
            return 'Normal'
        else:
            print(image_name)
            return None 

    df['label'] = df['file_name'].apply(lambda x: get_label(x, sus_path, normal_path))

    return df



def train_xgb_model(df, save_at):

    # if someone is NaN
    print(df['label'].isna().sum())

    X = df.drop(['label', 'file_name'], axis=1)  
    y = df['label'].map({'Suspicious': 0, 'Normal': 1})  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', objective='binary:logistic', tree_method='hist', eta=0.1, max_depth=10, enable_categorical=True)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the trained model
    model.save_model(save_at)




def classify(threeDresult, xgb_model, trained_keys):
    # first closest person
    keypoints = threeDresult['predictions'][0][0]['keypoints']
    angles = calculate_one_frame_angles(keypoints)

    df = pd.DataFrame([angles], columns=trained_keys)
    dmatrix = xgb.DMatrix(df)

    sus = xgb_model.predict(dmatrix)
    binary_prediction = (sus > 0.7).astype(int)[0]

    return binary_prediction


def draw_region(frame, binary_prediction):

    yolo_model = YOLO(r'./yolo11m-pose.pt')
    yolo_result = yolo_model(frame, verbose=False)[0]
    x1, y1, x2, y2 = yolo_result.boxes.xyxy[0].tolist()

    if binary_prediction == 0:
        conf_text = f'Suspicious'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, conf_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        conf_text = f'Normal'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (57, 255, 20), 2)
        cv2.putText(frame, conf_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (57, 255, 20), 2)




# --------------------------------------
# Function to process image and detect persons
def detect_persons_and_process(yolo_pt_path, image):
    model = YOLO(yolo_pt_path)  # Replace 'yolov8n.pt' with the specific YOLOv8 model you want
    # Run YOLOv8 inference
    results = model(image)
    # Filter results for 'person' class (class ID 0)
    persons = [result for result in results[0].boxes if result.cls == 0]
    return persons


def crop_persons(image, persons):
    crops = []
    for person in persons:
        x1, y1, x2, y2 = map(int, person.xyxy[0])  # Get bounding box coordinates
        crop = image[y1:y2, x1:x2]  # Crop the detected person
        crops.append((crop, (x1, y1, x2, y2)))
    return crops


def determine_suspects(inferencer, call_args, xgb_model, crops):
    susps = []

    for person in crops:
        call_args['inputs'] = person[0]

        for frame_3dresult in inferencer(**call_args):
            keypoints = frame_3dresult['predictions'][0][0]['keypoints']

            angles = calculate_one_frame_angles(keypoints)
            trained_keys = angles_keys()
            df = pd.DataFrame([angles], columns=trained_keys)
            dmatrix = xgb.DMatrix(df)

            sus = xgb_model.predict(dmatrix)
            binary_prediction = (sus > 0.7).astype(int)[0]

            susps.append(binary_prediction)

            # check_xgb(keypoints, sus, 0.7)

    return susps


# Function to draw bounding boxes and add header with color
def reconnect_and_draw(image, susps, crops):
    for suspect, (crop, (x1, y1, x2, y2)) in zip(susps, crops):
        if suspect == 1:
            # Suspect
            header = "Suspect"
            color = (0, 0, 255)  # Red
        else:
            # Normal
            header = "Normal"
            color = (0, 255, 0)  # Green

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(header, font, 0.5, 1)[0]
        text_x = x1
        text_y = y1 - 10

        # Draw header text
        cv2.putText(image, header, (text_x, text_y), font, 0.5, color, 1, cv2.LINE_AA)

        # Draw bounding box with specific color
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image



def plot_3d_pose(points):
    """
    Plots 17 points in 3D space as a pose with connections.
    
    Args:
    points (list of tuples): A list of 17 (x, y, z) coordinates.
    """
    if len(points) != 17:
        raise ValueError("The function expects exactly 17 points.")
    
    # Connections between points (index pairs)
    connections = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 5),
        (4, 6), (5, 7), (1, 8), (1, 9), (8, 10),
        (9, 11), (10, 12), (11, 13), (12, 14),
        (13, 15), (14, 16)
    ]
    
    # Unpack the points into x, y, z coordinates
    x, y, z = zip(*points)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(x, y, z, c='r', marker='o')
    
    # Plot the connections between points
    for start, end in connections:
        ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], color='b')
    
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Show the plot
    plt.show()



def check_xgb(keypoints, sus, threshold):
    
    x = keypoints[0::3]  # Extract every 3rd element starting from index 0 (x-coordinates)
    y = keypoints[1::3]  # Extract every 3rd element starting from index 1 (y-coordinates)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color='blue', label='Keypoints')
    plt.title('Keypoints Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # Calculate angles from keypoints
    angles = calculate_one_frame_angles(keypoints)
    trained_keys = angles_keys()

    # Bar chart for angles
    plt.figure(figsize=(8, 5))
    sns.barplot(x=trained_keys, y=angles, palette='viridis')
    plt.title('Angles Contributing to the Classification')
    plt.xlabel('Angle Features')
    plt.ylabel('Angle Values')
    plt.xticks(rotation=90)
    plt.show()

    binary_prediction = (sus > threshold).astype(int)[0]

    # Visualization of Model Output vs. Threshold
    plt.figure(figsize=(8, 5))
    plt.bar(['Model Output', 'Threshold'], [sus[0], threshold], color=['blue', 'red'])
    plt.title(f'Model Output vs. Threshold\nPrediction: {binary_prediction}')
    plt.ylabel('Prediction Value')
    plt.show()


# --------------------------------------

# x['visualization'][0]
def half_3d_img_result(img_np):

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    height, width, _ = img_bgr.shape
    cropped_image = img_bgr[:, width // 2:]

    return cropped_image

