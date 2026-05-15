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
if des_q is None or len(kp_q) < 2:
    print("Query image has too few SIFT features for matching")
    exit()

flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=3), dict(checks=40))

min_good_matches = 10
min_ransac_inliers = 8

qh, qw = query_img.shape
query_area = float(qw * qh)
min_area_ratio = 0.1
max_area_ratio = 1.0
query_corners = np.float32([[0, 0], [0, qh - 1], [qw - 1, qh - 1], [qw - 1, 0]]).reshape(-1, 1, 2)

fps = cap.get(cv.CAP_PROP_FPS) or 30.0
frame_delay = max(1, int(1000.0 / fps))

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
    status = ""

    if des_t is None or len(kp_t) < 2:
        status = "no features in frame (need SIFT descriptors)"
    elif len(good) < min_good_matches:
        status = f"not enough good matches: {len(good)}/{min_good_matches}"
    else:
        src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if H is None or mask is None:
            status = "homography failed (no robust fit)"
        else:
            inliers = int(mask.sum())
            if inliers < min_ransac_inliers:
                status = f"not enough inliers: {inliers}/{min_ransac_inliers}"
            else:
                new_corners = cv.perspectiveTransform(query_corners, H)
                proj_area = abs(cv.contourArea(new_corners))
                r = proj_area / query_area if query_area > 0 else 0.0
                if r < min_area_ratio or r > max_area_ratio:
                    new_corners = None
                    status = f"rejected pose: area ratio {r:.2f} (allowed {min_area_ratio}-{max_area_ratio})"
                else:
                    status = f" {len(good)} matches"

    if new_corners is not None:
        cv.polylines(frame, [np.int32(new_corners)], True, (0, 255, 0), 3, cv.LINE_AA)

    cv.putText(frame, status, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv.putText(frame, status, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


    result = cv.drawMatches(
        query_img, kp_q,
        frame, kp_t,
        good, None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


    cv.imshow('Feature Matching - Video', result)


    if cv.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()