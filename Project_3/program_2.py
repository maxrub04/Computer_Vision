import cv2 as cv
import numpy as np

query_img = cv.imread('photo_2_query.jpg')
train_img = cv.imread('photo_2_train.jpg')
if query_img is None or train_img is None:
    print("Image not found")
    exit()

query_gray = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
train_gray = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp_q, des_q = sift.detectAndCompute(query_gray, None)
kp_t, des_t = sift.detectAndCompute(train_gray, None)

flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
knn_matches = flann.knnMatch(des_q, des_t, k=2)

good = []
for m, n in knn_matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

train_out = train_img.copy()
matches_mask = None

if len(good) >= 10:
    src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    h, w = query_gray.shape
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    projected = cv.perspectiveTransform(corners, H)

    cv.polylines(train_out, [np.int32(projected)], True, (0, 255, 0), 3, cv.LINE_AA)
else:
    print(f"Not enough good matches: {len(good)}/10")

result = cv.drawMatches(
    query_img, kp_q,
    train_out, kp_t,
    good, None,
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv.imshow('Feature Matching', result)
cv.waitKey(0)
cv.destroyAllWindows()
