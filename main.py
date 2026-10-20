import argparse
import time
from pathlib import Path

import cv2
import numpy as np

VIDEOS_DIR = Path(__file__).parent / "test-videos"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_VIDEOS = ["mona-lisa.mp4", "mona-lisa-blur.mp4", "mona-lisa-blur-extra-credit.mp4"]
LABEL = "Mona Lisa"

TEMPLATE_SCALES = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18]
SIFT_CONTRAST = 0.01
RATIO = 0.75
MIN_INLIERS = 10
STRONG_INLIERS = 40
HOMOGRAPHY_METHOD = cv2.USAC_MAGSAC
RANSAC_THRESHOLD = 5.0
MIN_AREA = 100
FLANN_TREES = 5
FLANN_CHECKS = 50

FLOW_MAX_POINTS = 300
FLOW_QUALITY = 0.01
FLOW_MIN_DISTANCE = 4
FLOW_WINDOW = (21, 21)
FLOW_LEVELS = 3
FLOW_MAX_ERROR = 1.0
FLOW_RANSAC_THRESHOLD = 3.0
FLOW_RESEED_RATIO = 0.5

MIN_ANGLE = 35
MAX_SIDE_RATIO = 3.0
MAX_AREA_CHANGE = 1.6

BOX_COLOR = {"detect": (0, 255, 0), "flow": (0, 200, 255)}
TEXT_COLOR = (255, 255, 255)
BOX_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7


class Detector:
    def __init__(self, template_gray):
        self.sift = cv2.SIFT_create(contrastThreshold=SIFT_CONTRAST)
        self.matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": FLANN_TREES},
            {"checks": FLANN_CHECKS},
        )
        points, descriptors = [], []
        for scale in TEMPLATE_SCALES:
            image = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            keypoints, scale_descriptors = self.sift.detectAndCompute(image, None)
            if scale_descriptors is None:
                continue
            points += [(p.pt[0] / scale, p.pt[1] / scale) for p in keypoints]
            descriptors.append(scale_descriptors)
        self.points = np.float32(points)
        self.descriptors = np.vstack(descriptors)

    def find(self, gray):
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < MIN_INLIERS:
            return None, 0

        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        if len(good) < MIN_INLIERS:
            return None, 0

        source = self.points[[m.queryIdx for m in good]]
        target = np.float32([keypoints[m.trainIdx].pt for m in good])
        homography, mask = cv2.findHomography(source, target, HOMOGRAPHY_METHOD, RANSAC_THRESHOLD)
        if homography is None or mask.sum() < MIN_INLIERS:
            return None, 0
        return homography, int(mask.sum())


class Tracker:
    def __init__(self, template):
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        self.size = (width, height)
        self.corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        self.detector = Detector(gray)
        self.previous_gray = None
        self.homography = None
        self.quad = None
        self.points = None
        self.seeded = 0

    def locate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mode = "detect"
        homography, inliers = self.detector.find(gray)
        quad = self.project(homography, strict=inliers < STRONG_INLIERS)

        if (quad is None or inliers < STRONG_INLIERS) and self.points is not None:
            flow_homography = self.follow(gray)
            flow_quad = self.project(flow_homography)
            if flow_quad is not None:
                mode, homography, quad = "flow", flow_homography, flow_quad

        if quad is None:
            self.homography = self.quad = self.points = None
        else:
            self.homography, self.quad = homography, quad
            if mode == "detect" or len(self.points) < FLOW_RESEED_RATIO * self.seeded:
                self.seed(gray)

        self.previous_gray = gray
        return quad, mode

    def project(self, homography, strict=True):
        if homography is None:
            return None
        quad = cv2.perspectiveTransform(self.corners, homography)
        if not is_valid_quad(quad, self.size):
            return None
        if strict and self.quad is not None and not is_consistent(quad, self.quad):
            return None
        return quad

    def seed(self, gray):
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(self.quad.reshape(-1, 2)), 255)
        points = cv2.goodFeaturesToTrack(gray, FLOW_MAX_POINTS, FLOW_QUALITY, FLOW_MIN_DISTANCE, mask=mask)
        if points is None or len(points) < MIN_INLIERS:
            self.points = None
            return
        self.points = points.astype(np.float32)
        self.seeded = len(points)

    def follow(self, gray):
        params = {"winSize": FLOW_WINDOW, "maxLevel": FLOW_LEVELS}
        forward, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, self.points, None, **params)
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, forward, None, **params)
        error = np.linalg.norm((self.points - backward).reshape(-1, 2), axis=1)
        keep = (status.ravel() == 1) & (back_status.ravel() == 1) & (error < FLOW_MAX_ERROR)
        if keep.sum() < MIN_INLIERS:
            return None

        motion, mask = cv2.estimateAffine2D(
            self.points[keep], forward[keep], method=cv2.RANSAC, ransacReprojThreshold=FLOW_RANSAC_THRESHOLD
        )
        if motion is None or mask.sum() < MIN_INLIERS:
            return None

        self.points = forward[keep][mask.ravel() == 1]
        return np.vstack([motion, [0, 0, 1]]) @ self.homography


def is_valid_quad(quad, size):
    points = quad.reshape(-1, 2).astype(np.float64)
    if not cv2.isContourConvex(points.astype(np.float32)) or cv2.contourArea(points.astype(np.float32)) < MIN_AREA:
        return False

    width, height = size
    if np.any(points < [-width, -height]) or np.any(points > [2 * width, 2 * height]):
        return False

    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if max(sides[0], sides[2]) > MAX_SIDE_RATIO * min(sides[0], sides[2]):
        return False
    if max(sides[1], sides[3]) > MAX_SIDE_RATIO * min(sides[1], sides[3]):
        return False

    previous = np.roll(points, 1, axis=0) - points
    following = np.roll(points, -1, axis=0) - points
    cosines = np.sum(previous * following, axis=1) / (np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1))
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
    return bool(np.all(angles > MIN_ANGLE) and np.all(angles < 180 - MIN_ANGLE))


def is_consistent(quad, previous):
    area = cv2.contourArea(quad.reshape(-1, 2))
    previous_area = cv2.contourArea(previous.reshape(-1, 2))
    return previous_area / MAX_AREA_CHANGE < area < previous_area * MAX_AREA_CHANGE


def draw(frame, quad, label, mode):
    points = np.int32(quad)
    cv2.polylines(frame, [points], True, BOX_COLOR[mode], BOX_THICKNESS, cv2.LINE_AA)
    x, y = points.reshape(-1, 2)[np.argmax(quad.reshape(-1, 2)[:, 1])]
    (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 2)
    x = int(np.clip(x - text_width // 2, 0, frame.shape[1] - text_width))
    y = int(np.clip(y + text_height + 8, text_height, frame.shape[0] - baseline))
    cv2.putText(frame, label, (x, y), FONT, FONT_SCALE, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), FONT, FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA)


def process(video_path, label, show):
    capture = cv2.VideoCapture(str(video_path))
    ok, template = capture.read()
    if not ok:
        raise RuntimeError(f"Не удалось прочитать {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    height, width = template.shape[:2]
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"{video_path.stem}.mp4"
    output_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    tracker = Tracker(template)
    counts = {"detect": 0, "flow": 0, "lost": 0}
    timeline = ""
    started = time.perf_counter()

    frame = template
    while ok:
        quad, mode = tracker.locate(frame)
        if quad is None:
            mode = "lost"
        else:
            draw(frame, quad, label, mode)
        counts[mode] += 1
        timeline += {"detect": "#", "flow": "~", "lost": "."}[mode]
        writer.write(frame)
        if show:
            cv2.imshow(video_path.name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        ok, frame = capture.read()

    elapsed = time.perf_counter() - started
    capture.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    return {
        "video": video_path.name,
        "frames": len(timeline),
        "counts": counts,
        "timeline": timeline,
        "fps": len(timeline) / elapsed,
        "output": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Трекинг плоского объекта по ключевым точкам")
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--label", default=LABEL)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--timeline", action="store_true")
    args = parser.parse_args()

    videos = args.videos or [VIDEOS_DIR / name for name in DEFAULT_VIDEOS]
    for video in videos:
        stats = process(video, args.label, args.show)
        counts, frames = stats["counts"], stats["frames"]
        found = counts["detect"] + counts["flow"]
        print(
            f"{stats['video']}: найден в {found}/{frames} кадрах ({found / frames:.1%}): "
            f"детекция {counts['detect']}, оптический поток {counts['flow']}, "
            f"{stats['fps']:.1f} кадр/с -> {stats['output']}"
        )
        if args.timeline:
            line = stats["timeline"]
            for start in range(0, len(line), 60):
                print(f"  {start:4d} {line[start:start + 60]}")


if __name__ == "__main__":
    main()
