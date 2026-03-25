import cv2 as cv
import numpy as np

img = cv.imread('red_ball.jpg')
if img is None:
    print("Image not found")
    exit()

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
kernel = np.ones((5, 5), np.uint8)

mask0 = cv.inRange(hsv, np.array([0, 130, 70], np.uint8),   np.array([15, 255, 255], np.uint8))
mask1 = cv.inRange(hsv, np.array([165, 130, 70], np.uint8), np.array([179, 255, 255], np.uint8))

combined_mask = cv.bitwise_or(mask0, mask1)
combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)

contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if contours:
    largest = max(contours, key=cv.contourArea)
    cv.drawContours(img, [largest], -1, (255, 255, 255), 2)

    M = cv.moments(largest)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cv.circle(img, (cx, cy), 5, (255, 255, 255), -1)
    cv.putText(img, 'Red ball', (cx - 40, cy - 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
else:
    print("No red ball detected.")

cv.imshow('Red Ball', img)
cv.waitKey(0)