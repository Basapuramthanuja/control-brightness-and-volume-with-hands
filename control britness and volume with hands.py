import cv2
import mediapipe as mp
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

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

            # Get finger tips
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert to pixel coordinates
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            x_middle, y_middle = int(middle_tip.x * w), int(middle_tip.y * h)

            # Distance between thumb and index finger (for brightness)
            dist_thumb_index = np.linalg.norm(np.array([x_thumb, y_thumb]) - np.array([x_index, y_index]))
            brightness = np.interp(dist_thumb_index, [30, 200], [0, 100])
            sbc.set_brightness(int(brightness))

            # Distance between thumb and middle finger (for volume)
            dist_thumb_middle = np.linalg.norm(np.array([x_thumb, y_thumb]) - np.array([x_middle, y_middle]))
            volume_level = np.interp(dist_thumb_middle, [30, 200], [-65.25, 0])
            volume.SetMasterVolumeLevel(volume_level, None)

            # Display text on screen
            cv2.putText(frame, f'Brightness: {int(brightness)}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Volume: {int(np.interp(volume_level, [-65.25, 0], [0, 100]))}%', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw lines and circles for feedback
            cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (255, 0, 0), 3)
            cv2.circle(frame, (x_thumb, y_thumb), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)

            cv2.line(frame, (x_thumb, y_thumb), (x_middle, y_middle), (0, 0, 255), 3)
            cv2.circle(frame, (x_middle, y_middle), 10, (0, 0, 255), -1)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
