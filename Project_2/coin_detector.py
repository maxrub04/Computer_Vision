from pathlib import Path

import cv2 as cv
import numpy as np


def detect_tray_contour(image: np.ndarray) -> np.ndarray | None:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    lower_orange = np.array([5, 120, 120], dtype=np.uint8)
    upper_orange = np.array([12, 255, 255], dtype=np.uint8)
    mask = cv.inRange(hsv, lower_orange, upper_orange)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    tray = max(contours, key=cv.contourArea)
    if cv.contourArea(tray) < 0.1 * image.shape[0] * image.shape[1]:
        return None
    return cv.convexHull(tray)


def detect_coin_circles(image: np.ndarray) -> list[tuple[int, int, int]]:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=90,
        param2=47.5,
        minRadius=10,
        maxRadius=90,
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int)
    return [(int(x), int(y), int(r)) for x, y, r in circles]


def radius_threshold(radii: list[int]) -> float:
    if len(radii) < 2:
        return 32.0
    radii_arr = np.float32(radii).reshape(-1, 1)
    _, labels, centers = cv.kmeans(
        radii_arr,
        2,
        None,
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.1),
        10,
        cv.KMEANS_PP_CENTERS,
    )
    centers = np.sort(centers.flatten())
    if np.all(labels == labels[0]):
        return float(np.median(radii_arr))
    return float((centers[0] + centers[1]) / 2.0)


def annotate_and_count(image: np.ndarray, image_name: str) -> np.ndarray:
    output = image.copy()
    tray_contour = detect_tray_contour(image)
    tray_area = 0.0

    if tray_contour is not None:
        tray_area = cv.contourArea(tray_contour)
        cv.drawContours(output, [tray_contour], -1, (0, 255, 0), 3)

    circles = detect_coin_circles(image)
    radii = [r for _, _, r in circles]
    split_r = radius_threshold(radii)

    large_inside = 0
    large_outside = 0
    small_inside = 0
    small_outside = 0

    for x, y, r in circles:
        is_large = r >= split_r
        inside = (
            tray_contour is not None
            and cv.pointPolygonTest(tray_contour, (float(x), float(y)), False) >= 0
        )

        if is_large and inside:
            large_inside += 1
            color = (0, 0, 255)
        elif is_large and not inside:
            large_outside += 1
            color = (0, 0, 255)
        elif not is_large and inside:
            small_inside += 1
            color = (255, 0, 0)
        else:
            small_outside += 1
            color = (255, 0, 0)

        cv.circle(output, (x, y), r, color, 2)
        cv.circle(output, (x, y), 2, color, -1)

    lines = [
        f"Tray area: {tray_area} ",
        f"Large inside: {large_inside}",
        f"Large outside: {large_outside}",
        f"Small inside: {small_inside}",
        f"Small outside: {small_outside}",
    ]

    y0 = 28
    for i, text in enumerate(lines):
        y = y0 + i * 28
        cv.putText(output, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv.putText(output, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return output


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    image_paths = [base_dir / f"tray{i}.jpg" for i in range(1, 9)]

    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    for image_path in image_paths:
        image = cv.imread(str(image_path))
        if image is None:
            print(f"Skipping missing image: {image_path.name}")
            continue

        annotated = annotate_and_count(image, image_path.name)
        out_path = output_dir / f"annotated_{image_path.name}"
        cv.imwrite(str(out_path), annotated)
        cv.imshow(f"Coins - {image_path.name}", annotated)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()