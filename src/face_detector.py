"""
Face detection, tracking, and landmark extraction module.
Compatible with OpenCV 4.x (Haar + Facemark LBF) and OpenCV 5.x (YuNet DNN).
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from .face_swap_detector import FaceSwapDetector
    from .utils import crop_face_with_margin
except (ImportError, ValueError):
    from face_swap_detector import FaceSwapDetector
    from utils import crop_face_with_margin


class FaceTracker:
    """Tracks detected faces across frames and applies temporal EMA smoothing to bounding boxes."""

    def __init__(self, max_distance: float = 150.0, max_disappeared: int = 10, smooth_factor: float = 0.3):
        self.next_id = 1
        self.tracked_faces: Dict[int, Tuple[int, int, int, int]] = {}  # ID -> (center_x, center_y, width, height)
        self.disappeared: Dict[int, int] = {}                         # ID -> lost frame count
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.smooth_factor = smooth_factor

    def update(self, faces: List[Dict]) -> List[Dict]:
        """Matches faces to persistent IDs using centroid distance tracking."""
        if len(faces) == 0:
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    del self.tracked_faces[face_id]
                    del self.disappeared[face_id]
            return faces

        updated_tracked_faces = {}
        matched_ids = set()

        for face in faces:
            x, y, width, height = face["bbox"]
            center_x = x + width // 2
            center_y = y + height // 2

            best_id = None
            min_dist = float("inf")

            for face_id, (prev_x, prev_y, prev_w, prev_h) in self.tracked_faces.items():
                if face_id in matched_ids:
                    continue

                dist = math.hypot(center_x - prev_x, center_y - prev_y)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_id = face_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            else:
                # Smooth movement using Exponential Moving Average
                px, py, pw, ph = self.tracked_faces[best_id]
                center_x = int(px + self.smooth_factor * (center_x - px))
                center_y = int(py + self.smooth_factor * (center_y - py))
                width = int(pw + self.smooth_factor * (width - pw))
                height = int(ph + self.smooth_factor * (height - ph))

                face["bbox"] = (center_x - width // 2, center_y - height // 2, width, height)

            updated_tracked_faces[best_id] = (center_x, center_y, width, height)
            self.disappeared[best_id] = 0
            matched_ids.add(best_id)
            face["id"] = best_id
            face["face_id"] = best_id

        for face_id in list(self.tracked_faces.keys()):
            if face_id not in matched_ids:
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] <= self.max_disappeared:
                    updated_tracked_faces[face_id] = self.tracked_faces[face_id]
                else:
                    del self.disappeared[face_id]

        self.tracked_faces = updated_tracked_faces
        return faces


class FaceDetector:
    """
    Unified face detection and landmark extraction module.
    Automatically detects available OpenCV backends (YuNet DNN or Haar Cascade)
    and loads facial landmarks for forensic analysis.
    """

    def __init__(
        self,
        backend: str = "auto",
        scale_factor: float = 1.15,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (60, 60),
        margin: float = 0.25,
        confidence_threshold: float = 0.6,
        yunet_model_path: Optional[str] = None
    ):
        self.backend = backend.lower()
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.margin = margin
        self.confidence_threshold = confidence_threshold
        self.yunet_model_path = yunet_model_path

        self.tracker = FaceTracker()
        self.swap_detector = FaceSwapDetector()
        self.facemark = None

        self._init_backend()
        self._init_facemark()

    def _init_backend(self):
        """Initializes face detector backend with fallback safety."""
        project_root = Path(__file__).resolve().parent.parent

        # Candidate paths for YuNet ONNX
        candidate_paths = [
            project_root / "models" / "face_detection_yunet_2023mar.onnx",
            project_root / "models" / "face_detection_yunet.onnx",
            Path("models/face_detection_yunet_2023mar.onnx"),
            Path("models/face_detection_yunet.onnx"),
        ]
        if self.yunet_model_path:
            candidate_paths.insert(0, Path(self.yunet_model_path))

        resolved_yunet = None
        for p in candidate_paths:
            if p.is_file():
                resolved_yunet = str(p.resolve())
                break

        # 1. Prefer YuNet if model is present and cv2.FaceDetectorYN exists
        if hasattr(cv2, "FaceDetectorYN") and resolved_yunet:
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=resolved_yunet,
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.confidence_threshold,
                    nms_threshold=0.3,
                    top_k=5000
                )
                self.active_backend = "yunet"
                return
            except Exception as e:
                print(f"[FaceDetector] Warning: YuNet init failed: {e}")

        # 2. Try Haar Cascade if available
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            if not self.detector.empty():
                self.active_backend = "haar"
                return

        # 3. Fallback
        self.active_backend = "heuristic"

    def _init_facemark(self):
        """Initializes OpenCV Facemark LBF if available in opencv-contrib."""
        if hasattr(cv2, "face") and hasattr(cv2.face, "createFacemarkLBF"):
            try:
                facemark_model = Path(__file__).resolve().parent / "lbfmodel.yaml"
                if facemark_model.is_file():
                    self.facemark = cv2.face.createFacemarkLBF()
                    self.facemark.loadModel(str(facemark_model))
            except Exception as e:
                print(f"[FaceDetector] Warning: Facemark LBF model loading error: {e}")
                self.facemark = None

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detects faces in a frame, tracks their identity across time, extracts crops,
        and runs landmark and face-swap forensic analysis.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces: List[Dict] = []

        if self.active_backend == "yunet" and hasattr(self, "detector"):
            self.detector.setInputSize((w, h))
            _, detections = self.detector.detect(frame)
            if detections is not None:
                for f in detections:
                    conf = float(f[-1])
                    if conf >= self.confidence_threshold:
                        bx = max(0, int(f[0]))
                        by = max(0, int(f[1]))
                        bw = min(w - bx, int(f[2]))
                        bh = min(h - by, int(f[3]))
                        if bw >= self.min_size[0] and bh >= self.min_size[1]:
                            landmarks_5 = f[4:14].reshape((5, 2))
                            crop = frame[by:by + bh, bx:bx + bw].copy()
                            faces.append({
                                "bbox": (bx, by, bw, bh),
                                "crop": crop,
                                "confidence": conf,
                                "raw_landmarks": landmarks_5
                            })
        elif self.active_backend == "haar" and hasattr(self, "detector"):
            boxes = self.detector.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            for (bx, by, bw, bh) in boxes:
                crop = frame[by:by + bh, bx:bx + bw].copy()
                faces.append({
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "crop": crop,
                    "confidence": 0.95,
                    "raw_landmarks": None
                })
        else:
            # Heuristic detection for headless environments
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > (self.min_size[0] * self.min_size[1] * 1.5):
                    bx, by, bw, bh = cv2.boundingRect(c)
                    if 0.6 <= float(bw) / max(1, bh) <= 1.4:
                        crop = frame[by:by + bh, bx:bx + bw].copy()
                        faces.append({
                            "bbox": (bx, by, bw, bh),
                            "crop": crop,
                            "confidence": 0.70,
                            "raw_landmarks": None
                        })

        # Track faces across frames
        faces = self.tracker.update(faces)

        # Extract landmarks and enrich with crops & swap analysis
        for face in faces:
            bx, by, bw, bh = face["bbox"]
            face["raw_bbox"] = face["bbox"]

            # Add context margin crop for deep learning
            face_crop_expanded, expanded_box = crop_face_with_margin(frame, face["bbox"], self.margin)
            face["face_crop"] = face_crop_expanded
            face["expanded_bbox"] = expanded_box

            # Facemark 68-point landmarks if available
            landmarks_68 = []
            if self.facemark is not None and len(faces) > 0:
                try:
                    boxes_arr = np.array([[bx, by, bw, bh]])
                    success, landmarks_pts = self.facemark.fit(gray, boxes_arr)
                    if success and len(landmarks_pts) > 0:
                        pts = landmarks_pts[0][0]
                        landmarks_68 = [list(pt) for pt in pts]
                except Exception:
                    landmarks_68 = []

            face["landmarks"] = landmarks_68 if landmarks_68 else face.get("raw_landmarks", [])

            # Run face swap analysis
            face["swap_analysis"] = self.swap_detector.analyze_face(frame, face)

        return faces

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Convenience method returning (faces, frame)."""
        faces = self.detect_faces(frame)
        return faces, frame

    def annotate_frame(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and artifact annotations around detected faces."""
        output = frame.copy()

        for face in faces:
            x, y, width, height = face["bbox"]
            face_id = face.get("id", "?")
            analysis = face.get("swap_analysis")

            if analysis:
                score = analysis.get("score", 0.0)
                if analysis.get("quality") != "usable":
                    text = f"ID:{face_id} | Analysis limited"
                elif analysis.get("review_recommended"):
                    text = f"ID:{face_id} | Review artifacts ({score})"
                else:
                    text = f"ID:{face_id} | Artifact index: {score}"
            else:
                text = f"ID:{face_id} | Analysis unavailable"

            color = (0, 200, 255)
            thickness = 2
            length = int(width * 0.2)

            # Draw Corner brackets
            cv2.line(output, (x, y), (x + length, y), color, thickness)
            cv2.line(output, (x, y), (x, y + length), color, thickness)
            cv2.line(output, (x + width, y), (x + width - length, y), color, thickness)
            cv2.line(output, (x + width, y), (x + width, y + length), color, thickness)
            cv2.line(output, (x, y + height), (x + length, y + height), color, thickness)
            cv2.line(output, (x, y + height), (x, y + height - length), color, thickness)
            cv2.line(output, (x + width, y + height), (x + width - length, y + height), color, thickness)
            cv2.line(output, (x + width, y + height), (x + width, y + height - length), color, thickness)

            # Label text
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(text, font, 0.5, 2)
            cv2.rectangle(output, (x, max(y - text_h - 10, 0)), (x + text_w + 10, max(y, text_h + 10)), color, -1)
            cv2.putText(output, text, (x + 5, max(y - 5, text_h + 5)), font, 0.5, (0, 0, 0), 2)

        return output


def process_video(source: Union[int, str] = 0):
    """Detect faces from a webcam or video file."""
    detector = FaceDetector()
    video = cv2.VideoCapture(source)

    if not video.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            faces = detector.detect_faces(frame)
            output = detector.annotate_frame(frame, faces)

            cv2.putText(
                output,
                f"Faces detected: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("Face Detection", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index or path to video file")
    args = parser.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    process_video(src)
