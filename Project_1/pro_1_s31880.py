import cv2 as cv
import numpy as np

img = cv.imread('red_ball.jpg')
if img is None:
    print("Image not found")
    exit()

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
kernel = np.ones((5, 5), np.uint8)

lower_red = np.array([0, 150, 0], np.uint8)
upper_red = np.array([15, 255, 255], np.uint8)
mask0 = cv.inRange(hsv, lower_red, upper_red)

lower_red = np.array([165, 150, 70], np.uint8)
upper_red = np.array([179, 255, 255], np.uint8)
mask1 = cv.inRange(hsv, lower_red, upper_red)

combined_mask = cv.bitwise_or(mask0, mask1)
combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)

# --- Find contours on the cleaned mask ---
contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if contours:
    # Take the largest contour as the ball
    largest = max(contours, key=cv.contourArea)

    # Draw the contour outline
    cv.drawContours(img, [largest], -1, (255, 255, 255), 2)

    # Calculate centre of gravity using moments
    M = cv.moments(largest)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv.boundingRect(largest)
        cx, cy = x + w // 2, y + h // 2

    # Mark the centre of gravity
    cv.circle(img, (cx, cy), 5, (255, 255, 255), -1)

    # Draw text near the centre of gravity
    cv.putText(img, 'Red ball', (cx - 40, cy - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
else:
    print("No red ball detected.")

cv.imshow('Red Ball', img)
cv.waitKey(0)
cv.destroyAllWindows()