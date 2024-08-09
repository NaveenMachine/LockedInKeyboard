from operator import rshift
import cv2 as cv 
import numpy as np
import mediapipe as mp 
import pyautogui
import time

start_time = time.time()
last_click_x, last_click_y = pyautogui.position()


mpFaceMesh = mp.solutions.face_mesh
leftEye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
rightEye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] 
leftIris = [474, 475, 476, 477]
rightIris = [469, 470, 471, 472]

moveScaleX = 5000
moveScaleY = 5000

cap = cv.VideoCapture(0)
prevCenterLeft = None
prevCenterRight = None

with mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as faceMesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgbFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        imgH, imgW = frame.shape[:2]
        results = faceMesh.process(rgbFrame)
        if results.multi_face_landmarks:
            meshPoints = np.array([np.multiply([p.x, p.y], [imgW, imgH]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            cv.polylines(frame, [meshPoints[leftIris]], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [meshPoints[rightIris]], True, (0, 255, 0), 1, cv.LINE_AA)
            (lCx, lCy), lRadius = cv.minEnclosingCircle(meshPoints[leftIris])
            (rCx, rCy), rRadius = cv.minEnclosingCircle(meshPoints[rightIris])
            centerLeft = np.array([lCx, lCy], dtype=np.int32)
            centerRight = np.array([rCx, rCy], dtype=np.int32)
            cv.circle(frame, centerLeft, int(lRadius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, centerRight, int(rRadius), (255, 0, 255), 1, cv.LINE_AA)
            
            if prevCenterLeft is not None and prevCenterRight is not None:
                smoothCenterLeft = 0.9 * prevCenterLeft + 0.1 * centerLeft
                smoothCenterRight = 0.9 * prevCenterRight + 0.1 * centerRight
                dxLeft, dyLeft = smoothCenterLeft - prevCenterLeft
                dxRight, dyRight = smoothCenterRight - prevCenterRight
                pyautogui.move(moveScaleX*(dxLeft/50), moveScaleY*(dyLeft/50))
            
            prevCenterLeft = centerLeft
            prevCenterRight = centerRight
            
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        current_x, current_y = pyautogui.position()
        distance = ((current_x - last_click_x) ** 2 + (current_y - last_click_y) ** 2) ** 0.5  # Euclidean distance

        if distance <= 100:  # Check if within 100 pixels of last click position
            if time.time() - start_time >= 1.5:
                pyautogui.click()
                start_time = time.time()
        else:
            last_click_x, last_click_y = current_x, current_y
            start_time = time.time()

        # Check every 0.1 second
        time.sleep(0.005)

cap.release()
cv.destroyAllWindows()