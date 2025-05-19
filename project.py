import cv2
import numpy as np

# Load video from file instead of webcam
cap = cv2.VideoCapture("smoke.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('.mp4', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire color range (red hues)
    lower_fire = np.array([0, 150, 150])
    upper_fire = np.array([15, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
    fire_detected = cv2.bitwise_and(frame, frame, mask=fire_mask)

    # Smoke color range (grayish hues)
    lower_smoke = np.array([90, 0, 50])
    upper_smoke = np.array([110, 50, 200])
    smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
    smoke_detected = cv2.bitwise_and(frame, frame, mask=smoke_mask)

    # Overlay detection results
    cv2.putText(frame, 'Fire & Smoke Detection', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if np.any(fire_mask):
        cv2.putText(frame, 'FIRE DETECTED!', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if np.any(smoke_mask):
        cv2.putText(frame, 'SMOKE DETECTED!', (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Fire Detection', fire_detected)
    cv2.imshow('Smoke Detection', smoke_detected)

    out.write(frame)  # Save the processed frame to output video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()