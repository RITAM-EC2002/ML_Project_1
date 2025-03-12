# Packages
import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            
            # Get coordinates of thumb tip and index finger tip
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
            
            # Draw a line between thumb and index finger
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # Calculate distance
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Normalize distance for volume and brightness control
            min_dist = 30
            max_dist = 200
            normalized_dist = np.clip((distance - min_dist) / (max_dist - min_dist), 0, 1)
            
            # Adjust volume and brightness
            volume_level = normalized_dist * (volume.GetVolumeRange()[1] - volume.GetVolumeRange()[0]) + volume.GetVolumeRange()[0]
            volume.SetMasterVolumeLevel(volume_level, None)
            brightness_level = int(normalized_dist * 100)
            sbc.set_brightness(brightness_level)
            
            # Display volume and brightness level
            cv2.putText(frame, f'Volume: {int(normalized_dist * 100)}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Brightness: {brightness_level}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
