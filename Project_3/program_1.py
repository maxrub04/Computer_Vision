import cv2 as cv
import numpy as np

img = cv.imread('photo_1.jpg')
if img is None:
    print("Image not found")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_float = np.float32(gray)

harris = cv.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

local_max = cv.dilate(harris, np.ones((15, 15), np.uint8))
peaks = (harris == local_max) & (harris > 0.01 * harris.max())

ys, xs = np.where(peaks)
scores = harris[ys, xs]
order = np.argsort(-scores)

min_dist = 40
chosen = []

for idx in order:
    y, x = ys[idx], xs[idx]
    too_close = False
    for cx, cy in chosen:
        if (x - cx) ** 2 + (y - cy) ** 2 < min_dist ** 2:
            too_close = True
            break
    if not too_close:
        chosen.append((x, y))
    if len(chosen) == 4:
        break

harris_img = img.copy()
for (x, y) in chosen:
    cv.circle(harris_img, (x, y), 12, (0, 0, 255), 2)
    cv.circle(harris_img, (x, y), 2, (0, 0, 255), -1)

sift = cv.SIFT_create(nfeatures=0, contrastThreshold=0.09, edgeThreshold=15, sigma=3)
keypoints, _ = sift.detectAndCompute(gray, None)

sift_img = img.copy()
cv.drawKeypoints(sift_img, keypoints, sift_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('Original', img)
cv.imshow('Harris', harris_img)
cv.imshow('SIFT', sift_img)
cv.waitKey(0)
cv.destroyAllWindows()
