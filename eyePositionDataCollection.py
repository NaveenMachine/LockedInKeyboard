import cv2
import numpy as np
import dlib
from imutils import face_utils
import csv
import time

# Load the face detector and face landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to calculate the 3D gaze direction
def get_gaze_direction(shape):
    left_eye_pts = shape[36:42]
    right_eye_pts = shape[42:48]

    left_eye_center = left_eye_pts.mean(axis=0).astype(int)
    right_eye_center = right_eye_pts.mean(axis=0).astype(int)

    return left_eye_center, right_eye_center

# Create and open a CSV file for writing
with open('eye_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Left Eye X', 'Left Eye Y', 'Right Eye X', 'Right Eye Y'])

    # Start the webcam feed
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye_center, right_eye_center = get_gaze_direction(shape)

            # Write the data to the CSV file
            timestamp = time.time()
            writer.writerow([timestamp, left_eye_center[0], left_eye_center[1], right_eye_center[0], right_eye_center[1]])

            cv2.circle(frame, tuple(left_eye_center), 3, (0, 255, 0), -1)
            cv2.circle(frame, tuple(right_eye_center), 3, (0, 255, 0), -1)

            # Draw the eye regions
            left_eye_hull = cv2.convexHull(shape[36:42])
            right_eye_hull = cv2.convexHull(shape[42:48])
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            # For demonstration, draw the line from the eye center to show gaze direction (simplified)
            cv2.line(frame, tuple(left_eye_center), (left_eye_center[0], left_eye_center[1] - 50), (255, 0, 0), 2)
            cv2.line(frame, tuple(right_eye_center), (right_eye_center[0], right_eye_center[1] - 50), (255, 0, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
