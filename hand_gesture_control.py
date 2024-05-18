import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2 + (point2.z - point1.z) ** 2)

# Function to check if a fist is open or closed
def is_fist_open(landmarks):
    # Fist open detection by checking if all fingertips are above their respective PIP joints
    fingers = [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]
    
    for tip, pip in fingers:
        if landmarks[tip].y >= landmarks[pip].y:
            return False
    return True

# Function to check if the thumb is closed (touching the palm)
def is_thumb_closed(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return calculate_distance(thumb_tip, index_mcp) < 0.05  # Adjust threshold as needed

# Function to control brightness
def adjust_brightness(up):
    current_brightness = sbc.get_brightness(display=0)[0]  # Get the first element of the list
    if up:
        new_brightness = min(current_brightness + 10, 100)
    else:
        new_brightness = max(current_brightness - 10, 0)
    sbc.set_brightness(new_brightness, display=0)

# Function to initialize the audio interface
def get_audio_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume

# Function to control volume
def adjust_volume(volume, up):
    current_volume = volume.GetMasterVolumeLevelScalar()
    step = 0.1  # Volume step (10%)
    if up:
        new_volume = min(current_volume + step, 1.0)
    else:
        new_volume = max(current_volume - step, 0.0)
    volume.SetMasterVolumeLevelScalar(new_volume, None)

# Initialize the audio interface
audio_interface = get_audio_interface()

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for natural hand movement
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get hand landmarks
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the landmarks for the thumb tip and index finger tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate the distance between the thumb tip and index finger tip
            distance = calculate_distance(thumb_tip, index_finger_tip)
            
            # Adjust the volume based on the distance between thumb and index finger
            if distance > 0.1:  # Adjust this threshold based on your needs
                adjust_volume(audio_interface, True)
            elif distance < 0.05:  # Adjust this threshold based on your needs
                adjust_volume(audio_interface, False)
            
            # Check for the fist open/close gesture and adjust brightness accordingly
            if is_fist_open(hand_landmarks.landmark):
                adjust_brightness(True)
            else:
                adjust_brightness(False)
            
            # Check if the thumb is closed to adjust volume
            if is_thumb_closed(hand_landmarks.landmark):
                adjust_volume(audio_interface, False)
    
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
