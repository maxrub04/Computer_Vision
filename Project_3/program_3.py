import cv2 as cv
import numpy as np

query_img = cv.imread('photo_3_query.jpg', cv.IMREAD_GRAYSCALE)
if query_img is None:
    print("Query image not found")
    exit()

cap = cv.VideoCapture('video_3_train.mp4')
if not cap.isOpened():
    print("Video not found")
    exit()

sift = cv.SIFT_create()
kp_q, des_q = sift.detectAndCompute(query_img, None)

flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

qh, qw = query_img.shape
query_corners = np.float32([[0, 0], [0, qh - 1], [qw - 1, qh - 1], [qw - 1, 0]]).reshape(-1, 1, 2)

fps = cap.get(cv.CAP_PROP_FPS) or 30.0
frame_delay = max(1, int(1000.0 / fps))

last_corners = None
smoothed = None
alpha = 0.35
max_jump = 80

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp_t, des_t = sift.detectAndCompute(gray, None)

    good = []
    if des_t is not None and len(kp_t) >= 2:
        knn_matches = flann.knnMatch(des_q, des_t, k=2)
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    new_corners = None
    inliers = 0
    inlier_mask = None

    if len(good) >= 10:
        src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if H is not None and mask is not None:
            inliers = int(mask.sum())
            inlier_mask = mask.ravel()

            if inliers >= 8:
                projected = cv.perspectiveTransform(query_corners, H)

                if smoothed is None:
                    new_corners = projected
                else:
                    diff = np.linalg.norm(projected - smoothed, axis=2).max()
                    if diff < max_jump:
                        new_corners = projected

    if new_corners is not None:
        if smoothed is None:
            smoothed = new_corners
        else:
            smoothed = alpha * new_corners + (1.0 - alpha) * smoothed

        last_corners = smoothed
        color = (0, 255, 0)
        status = f"matches: {len(good)}  inliers: {inliers}"

        if inlier_mask is not None:
            for i, m in enumerate(good):
                if inlier_mask[i]:
                    pt = kp_t[m.trainIdx].pt
                    cv.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

    elif last_corners is not None:
        color = (0, 165, 255)
        status = f"last good position (matches: {len(good)})"
    else:
        color = None
        status = f"searching... (matches: {len(good)})"

    if last_corners is not None and color is not None:
        cv.polylines(frame, [np.int32(last_corners)], True, color, 3, cv.LINE_AA)

    cv.putText(frame, status, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv.putText(frame, status, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv.imshow('Feature Matching - Video', frame)
    if cv.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
