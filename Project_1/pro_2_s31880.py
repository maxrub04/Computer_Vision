import cv2 as cv
import numpy as np

cap = cv.VideoCapture('rgb_ball_720.mp4')
if not cap.isOpened():
    print("Video not found")
    exit()

kernel = np.ones((5, 5), np.uint8)
last_cx, last_cy = None, None

while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask0 = cv.inRange(hsv, np.array([0, 150, 0], np.uint8),    np.array([15, 255, 255], np.uint8))
    mask1 = cv.inRange(hsv, np.array([165, 150, 70], np.uint8), np.array([179, 255, 255], np.uint8))

    combined_mask = cv.bitwise_or(mask0, mask1)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)

    contours, i = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv.contourArea)
        M = cv.moments(largest)
        if M["m00"] != 0:
            last_cx = int(M["m10"] / M["m00"])
            last_cy = int(M["m01"] / M["m00"])

            cv.drawContours(frame, [largest], -1, (255, 255, 255), 2)
            cv.circle(frame, (last_cx, last_cy), 5, (255, 255, 255), -1)
            cv.putText(frame, 'Red ball', (last_cx - 40, last_cy - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif last_cx is not None:
        cv.circle(frame, (last_cx, last_cy), 30, (200, 200, 200), 2)
        cv.circle(frame, (last_cx, last_cy), 5, (200, 200, 200), -1)
        cv.putText(frame, 'Red ball (last seen)', (last_cx - 60, last_cy - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv.imshow('Red Ball', frame)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()