"""
Face detection and tracking module for real-time video frames.
Supports OpenCV YuNet DNN detector with automatic model resolution and fallback mechanisms.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .utils import crop_face_with_margin


class FaceDetector:
    """
    Detects faces in video frames and assigns persistent tracking IDs using IoU matching.
    Provides cropped face images with context margins for deepfake forensic inspection.
    """

    def __init__(
        self,
        backend: str = "yunet",
        min_size: Tuple[int, int] = (40, 40),
        confidence_threshold: float = 0.6,
        margin: float = 0.25,
        yunet_model_path: Optional[str] = None
    ):
        """
        Args:
            backend: 'yunet' (high-accuracy DNN) or 'haar' / 'fallback'.
            min_size: Minimum face detection dimensions (width, height).
            confidence_threshold: Confidence threshold for detection.
            margin: Extra context percentage to crop around face borders.
            yunet_model_path: Path to face_detection_yunet_*.onnx model file.
        """
        self.backend = backend.lower()
        self.min_size = min_size
        self.confidence_threshold = confidence_threshold
        self.margin = margin
        self.yunet_model_path = yunet_model_path

        # Persistent tracker state: face_id -> last_bbox
        self.active_tracks: Dict[int, Tuple[int, int, int, int]] = {}
        self.next_face_id: int = 0

        # Initialize detector backend
        self._init_backend()

    def _init_backend(self):
        """Initializes face detector backend with fallback safety."""
        # 1. Try finding YuNet ONNX model
        candidate_paths = []
        if self.yunet_model_path:
            candidate_paths.append(Path(self.yunet_model_path))

        # Check default project locations
        project_root = Path(__file__).resolve().parent.parent
        candidate_paths.extend([
            project_root / "models" / "face_detection_yunet_2023mar.onnx",
            project_root / "models" / "face_detection_yunet.onnx",
            Path("models/face_detection_yunet_2023mar.onnx"),
            Path("models/face_detection_yunet.onnx"),
        ])

        resolved_model = None
        for p in candidate_paths:
            if p.is_file():
                resolved_model = str(p.resolve())
                break

        if hasattr(cv2, "FaceDetectorYN") and resolved_model:
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=resolved_model,
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.confidence_threshold,
                    nms_threshold=0.3,
                    top_k=5000
                )
                self.backend = "yunet"
                return
            except Exception as e:
                print(f"[FaceDetector] Warning: YuNet init failed ({e}). Falling back to alternate detector.")

        # 2. Try Haar Cascade if available in this OpenCV build
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if not self.cascade.empty():
                self.backend = "haar"
                return

        # 3. Built-in skin/contour heuristic fallback (guarantees zero crash in any environment)
        self.backend = "heuristic"

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Finds all faces in a frame, tracks their identity across frames, and extracts crops.

        Args:
            frame_bgr: BGR video frame from camera or file.

        Returns:
            List of detection dictionaries.
        """
        h, w = frame_bgr.shape[:2]
        raw_detections: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray]]] = []

        if self.backend == "yunet" and hasattr(self, "detector"):
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame_bgr)
            if faces is not None:
                for f in faces:
                    confidence = float(f[-1])
                    if confidence >= self.confidence_threshold:
                        box = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
                        # Ensure within bounds
                        box = (
                            max(0, box[0]),
                            max(0, box[1]),
                            min(w - max(0, box[0]), box[2]),
                            min(h - max(0, box[1]), box[3])
                        )
                        if box[2] >= self.min_size[0] and box[3] >= self.min_size[1]:
                            landmarks = f[4:14].reshape((5, 2))
                            raw_detections.append((box, confidence, landmarks))
        elif self.backend == "haar" and hasattr(self, "cascade"):
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, fw, fh) in faces:
                raw_detections.append(((x, y, fw, fh), 0.95, None))
        else:
            # Heuristic detector for synthetic tests or headless setups
            raw_detections = self._heuristic_detect(frame_bgr)

        # Associate detections with tracking IDs using Intersection over Union (IoU)
        matched_results = self._match_tracks(raw_detections, frame_bgr)
        return matched_results

    def _heuristic_detect(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray]]]:
        """Fallback face detector based on skin chromaticity contours."""
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Skin color range in HSV
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > (self.min_size[0] * self.min_size[1] * 1.5):
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect_ratio = float(bw) / max(1, bh)
                if 0.6 <= aspect_ratio <= 1.4:
                    detections.append(((bx, by, bw, bh), 0.70, None))
        return detections

    def _match_tracks(
        self,
        raw_detections: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray]]],
        frame_bgr: np.ndarray
    ) -> List[Dict]:
        """Matches current detections to existing track IDs to prevent flicker."""
        results: List[Dict] = []
        updated_tracks: Dict[int, Tuple[int, int, int, int]] = {}

        unmatched_detections = list(enumerate(raw_detections))

        # Match against existing tracks using highest IoU
        for track_id, prev_box in list(self.active_tracks.items()):
            best_iou = 0.0
            best_idx = -1
            best_det = None

            for i, (orig_idx, (box, conf, landmarks)) in enumerate(unmatched_detections):
                iou = self._calculate_iou(prev_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
                    best_det = (box, conf, landmarks)

            if best_iou > 0.30 and best_det is not None:
                box, conf, landmarks = best_det
                updated_tracks[track_id] = box
                del unmatched_detections[best_idx]

                face_crop, expanded_box = crop_face_with_margin(frame_bgr, box, self.margin)
                results.append({
                    "face_id": track_id,
                    "raw_bbox": box,
                    "bbox": expanded_box,
                    "face_crop": face_crop,
                    "confidence": conf,
                    "landmarks": landmarks
                })

        # New detections become new tracks
        for _, (box, conf, landmarks) in unmatched_detections:
            new_id = self.next_face_id
            self.next_face_id += 1
            updated_tracks[new_id] = box

            face_crop, expanded_box = crop_face_with_margin(frame_bgr, box, self.margin)
            results.append({
                "face_id": new_id,
                "raw_bbox": box,
                "bbox": expanded_box,
                "face_crop": face_crop,
                "confidence": conf,
                "landmarks": landmarks
            })

        self.active_tracks = updated_tracks
        return results

    @staticmethod
    def _calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Calculates Intersection over Union between two (x, y, w, h) boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        areaA = boxA[2] * boxA[3]
        areaB = boxB[2] * boxB[3]
        union_area = float(areaA + areaB - inter_area)

        if union_area <= 0:
            return 0.0
        return inter_area / union_area
