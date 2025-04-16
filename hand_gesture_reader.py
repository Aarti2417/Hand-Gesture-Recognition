# Author: Odilbek Tokhirov
# Opens the webcam, predicts the gesture shown, activates arrow keys or sends gesture to serial port
# Press Q to close the webcam

import mediapipe as mp
import cv2
import numpy as np
import joblib
import pyautogui as pag
import serial
import time
import argparse
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)

# Command-line arguments for model selection
parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
parser.add_argument('--model', type=str, default='model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl', help='Path to the model file')
args = parser.parse_args()

# Load the model
try:
    model = joblib.load(args.model)
    logging.info(f"Loaded model: {args.model}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    exit()

# MediaPipe and gesture mappings
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
idx_to_class = {
    0: 'Closed',
    1: 'Three',
    2: 'Open',
    3: 'Zero',
}
class_to_key = {
    'Closed': 'up',
    'Three': 'right',
    'Open': 'left',
    'Zero': 'down',
}

# Serial communication setup
try:
    serial_port = serial.Serial('COM7', 9600, timeout=1)
    logging.info("Serial port opened successfully")
except Exception as e:
    logging.error(f"Failed to open serial port: {e}")
    serial_port = None

# Gesture detection and prediction
def detect_gesture(image, hands):
    """Detect hands and return landmarks."""
    results = hands.process(image)
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                landmarks.extend([lm.x, lm.y])
    return np.array(landmarks)[None, :] if landmarks else None

def predict_gesture(model, landmarks):
    """Predict gesture from landmarks."""
    try:
        if landmarks is None or len(landmarks[0]) != model.n_features_in_:
            logging.warning("Invalid input shape for prediction")
            return None
        yhat_idx = int(model.predict(landmarks)[0])
        return idx_to_class[yhat_idx]
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

def send_to_serial(serial_port, command):
    """Send command to serial port."""
    try:
        if serial_port and serial_port.is_open:
            serial_port.write(f'{command}\n'.encode())
            logging.info(f"Sent command to serial port: {command}")
    except Exception as e:
        logging.error(f"Failed to send command to serial port: {e}")

# Main loop
def main():
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.4)
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        logging.error("Failed to open webcam")
        return

    current_command = None
    last_sent_time = 0
    send_interval = 1  # Minimum time between sends in seconds

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                logging.error("Failed to read frame from webcam")
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]

            landmarks = detect_gesture(image, hands)
            yhat = predict_gesture(model, landmarks)

            if yhat:
                x_max = int(width * np.max(landmarks[0][::2]))
                x_min = int(width * np.min(landmarks[0][::2]))
                y_max = int(height * np.max(landmarks[0][1::2]))
                y_min = int(height * np.min(landmarks[0][1::2]))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
                cv2.putText(frame, f'{yhat}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                current_time = time.time()
                if yhat != current_command and (current_time - last_sent_time > send_interval):
                    pag.press(class_to_key[yhat])  # Perform the action
                    send_to_serial(serial_port, yhat)  # Send gesture to serial port
                    current_command = yhat
                    last_sent_time = current_time
            else:
                current_command = None

            cv2.imshow('Hand Gesture Reader', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.release()
        if serial_port and serial_port.is_open:
            serial_port.close()
        cv2.destroyAllWindows()
        logging.info("Resources released successfully")

if __name__ == "__main__":
    main()
