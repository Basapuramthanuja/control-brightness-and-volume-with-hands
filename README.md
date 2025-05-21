🖐️ Hand Gesture Based Brightness and Volume Control

This project lets you control your **system volume** and **screen brightness** using **hand gestures**. It uses your **webcam** to detect finger movements in real-time with the help of **MediaPipe** and controls your system using **pycaw** and **screen-brightness-control** libraries.
## 🎯 Objective

To build a smart, contactless control system that uses hand gestures to manage system volume and brightness.
## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV** – for webcam and image processing
- **MediaPipe** – for real-time hand tracking
- **pycaw** – for volume control (Windows)
- **screen-brightness-control** – for brightness
- **numpy, math** – for calculations

---

## ✅ Step-by-Step Process (How I Did It)

### 1️⃣ Project Setup

- Opened **VS Code**
- Created a folder: `gesture-control`
- Created a file: `main.py`

---

### 2️⃣ Installed Required Libraries

In the terminal, run:
pip install opencv-python mediapipe pycaw screen-brightness-control numpy
Accessed the Webcam & Hand Detection
Used OpenCV for webcam and MediaPipe for hand tracking:
4️⃣ Extracted Finger Landmarks
python
Copy
Edit
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
Extracted Finger Landmarks
Used MediaPipe to get landmark positions of fingers (e.g., thumb tip, index tip):

python
Copy
Edit
for lm in handLms.landmark:
    h, w, c = img.shape
    cx, cy = int(lm.x * w), int(lm.y * h)
5️⃣ Calculated Distance Between Fingers
Used math.hypot() to calculate:

Thumb ↔ Index → Volume

Index ↔ Middle → Brightness

python
Copy
Edit
import math
length = math.hypot(x2 - x1, y2 - y1)
6️⃣ Controlled System Volume
Used pycaw for volume control:

python
Copy
Edit
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Map finger distance to volume (0.0 to 1.0)
volume.SetMasterVolumeLevelScalar(vol, None)
