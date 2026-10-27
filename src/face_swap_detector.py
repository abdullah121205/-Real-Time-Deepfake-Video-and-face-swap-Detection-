"""Conservative, interpretable face-swap artifact screening and deepfake forensics.

Supports both:
1. Multi-frame landmark-defined boundary anomaly estimation (from face_analysis_module).
2. Direct face crop spectral (2D-FFT), boundary gradient, and chromaticity analysis (from VideoProcessor).
"""

from collections import defaultdict, deque
from typing import Dict, Optional, Union
import cv2
import numpy as np


class FaceSwapDetector:
    """Estimate visual anomalies in face regions and boundaries."""

    def __init__(self, history_length=12, min_persistent_frames=6, smooth_factor=0.25):
        self.history_length = max(4, int(history_length))
        self.min_persistent_frames = max(2, min(int(min_persistent_frames), self.history_length))
        self.smooth_factor = float(np.clip(smooth_factor, 0.0, 1.0))
        self.history = defaultdict(lambda: deque(maxlen=self.history_length))
        self.last_score = {}

    @staticmethod
    def _result(details, score=0.0, quality="insufficient", signals=None,
                review_recommended=False, stable_frames=0):
        norm_score = float(np.clip(score / 100.0 if score > 1.0 else score, 0.0, 1.0))
        signals = signals or {}
        return {
            "score": round(float(score), 1),
            "swap_score": norm_score,
            "boundary_anomaly": signals.get("boundary_edge_ratio", norm_score),
            "frequency_anomaly": signals.get("relative_texture_difference", norm_score),
            "color_disparity": signals.get("chroma_signal_0_to_1", norm_score),
            "is_fake": None,
            "review_recommended": bool(review_recommended),
            "quality": quality,
            "stable_frames": int(stable_frames),
            "details": details,
            "signals": signals,
        }

    @staticmethod
    def _laplacian_variance(gray, mask):
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        values = lap[mask > 0]
        if values.size < 25:
            return 0.0
        return float(np.var(values))

    def analyze_face(self, frame_or_crop: np.ndarray, face_data: Optional[Dict] = None) -> Dict:
        """
        Analyzes a face for face-swap artifacts.
        Can be called with:
            analyze_face(crop_bgr) -> direct crop analysis
            analyze_face(frame_bgr, face_data) -> full-frame landmark analysis
        """
        if frame_or_crop is None or not isinstance(frame_or_crop, np.ndarray) or frame_or_crop.size == 0:
            return self._result("Invalid frame or crop")

        # Mode A: Full frame with landmark annotations provided
        if face_data is not None:
            return self._analyze_with_landmarks(frame_or_crop, face_data)

        # Mode B: Direct face crop analysis (used by VideoProcessor and Unit Tests)
        return self._analyze_direct_crop(frame_or_crop)

    def _analyze_direct_crop(self, crop_bgr: np.ndarray) -> Dict:
        """Analyzes a cropped face region using FFT spectral power and edge gradients."""
        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 20:
            return self._result("Crop too small", score=0.0)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Edge & gradient boundary discontinuity
        outer_mask = np.ones((h, w), dtype=np.uint8)
        inner_y1, inner_y2 = int(h * 0.20), int(h * 0.80)
        inner_x1, inner_x2 = int(w * 0.20), int(w * 0.80)
        outer_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)

        outer_mag_mean = float(np.mean(mag[outer_mask == 1])) + 1e-6
        inner_mag_mean = float(np.mean(mag[outer_mask == 0])) + 1e-6
        ratio = outer_mag_mean / inner_mag_mean
        boundary_anomaly = float(1.0 / (1.0 + np.exp(-3.0 * (abs(ratio - 1.0) - 0.5))))

        # 2. 2D FFT Frequency Analysis
        resized = cv2.resize(gray, (128, 128)).astype(np.float32)
        dft = np.fft.fft2(resized)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.log(np.abs(dft_shift) + 1.0)

        y, x = np.ogrid[:128, :128]
        dist_from_center = np.sqrt((x - 64) ** 2 + (y - 64) ** 2)
        low_power = float(np.mean(magnitude[dist_from_center <= 24])) + 1e-6
        high_power = float(np.mean(magnitude[dist_from_center > 40])) + 1e-6
        hf_ratio = high_power / low_power
        freq_anomaly = float(1.0 / (1.0 + np.exp(-12.0 * (abs(hf_ratio - 0.62) - 0.12))))

        # 3. LAB Chroma Disparity
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        inner = lab[inner_y1:inner_y2, inner_x1:inner_x2]
        inner_a = float(np.mean(inner[:, :, 1]))
        inner_b = float(np.mean(inner[:, :, 2]))
        outer_a = float(np.mean(lab[:, :, 1][outer_mask == 1]))
        outer_b = float(np.mean(lab[:, :, 2][outer_mask == 1]))
        chroma_dist = np.sqrt((inner_a - outer_a) ** 2 + (inner_b - outer_b) ** 2)
        color_disparity = float(1.0 / (1.0 + np.exp(-5.0 * (chroma_dist / 40.0 - 0.35))))

        composite = 0.40 * boundary_anomaly + 0.35 * freq_anomaly + 0.25 * color_disparity
        composite = float(np.clip(composite, 0.0, 1.0))

        signals = {
            "boundary_edge_ratio": round(boundary_anomaly, 3),
            "relative_texture_difference": round(freq_anomaly, 3),
            "chroma_signal_0_to_1": round(color_disparity, 3),
        }

        return {
            "score": round(composite * 100.0, 1),
            "swap_score": composite,
            "boundary_anomaly": boundary_anomaly,
            "frequency_anomaly": freq_anomaly,
            "color_disparity": color_disparity,
            "is_fake": None,
            "review_recommended": composite >= 0.60,
            "quality": "usable",
            "stable_frames": 1,
            "details": "Direct crop frequency and edge analysis",
            "signals": signals
        }

    def _analyze_with_landmarks(self, frame: np.ndarray, face_data: Dict) -> Dict:
        """Original landmark-based screening algorithm from PR #1."""
        landmarks = face_data.get("landmarks", [])
        if landmarks is None or len(landmarks) < 68:
            # If 68-landmarks unavailable, fall back to crop analysis
            crop = face_data.get("crop")
            if crop is not None:
                return self._analyze_direct_crop(crop)
            return self._result("Landmarks unavailable; no assessment made")

        try:
            x, y, w, h = [int(v) for v in face_data["bbox"]]
        except (KeyError, TypeError, ValueError):
            return self._result("Invalid face bounding box")
        if w < 80 or h < 80:
            return self._result("Face region too small for meaningful analysis")

        frame_h, frame_w = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
        if x2 <= x1 or y2 <= y1:
            return self._result("Face region lies outside the frame")

        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] < 60 or roi.shape[1] < 60:
            return self._result("Clipped face region too small")

        try:
            pts_global = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        except (TypeError, ValueError):
            return self._result("Invalid landmark coordinates")
        if pts_global.shape[0] < 68 or not np.isfinite(pts_global[:68]).all():
            return self._result("Invalid landmark coordinates")
        pts_global = pts_global[:68]
        inside = (
            (pts_global[:, 0] >= x1) & (pts_global[:, 0] < x2) &
            (pts_global[:, 1] >= y1) & (pts_global[:, 1] < y2)
        )
        if float(np.mean(inside)) < 0.90:
            return self._result("Landmarks do not align with the face region")

        points = np.rint(pts_global - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
        hull = cv2.convexHull(points)
        face_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, hull, 255)
        face_area = cv2.countNonZero(face_mask)
        if face_area < 500 or face_area > roi.shape[0] * roi.shape[1] * 0.98:
            return self._result("Unreliable landmark face mask")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        if blur < 18:
            return self._result("Low-detail or blurred face; assessment skipped", quality="low")
        if brightness < 25 or brightness > 232:
            return self._result("Very dark or overexposed face; assessment skipped", quality="low")

        k = max(3, (min(roi.shape[:2]) // 20) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        inner_mask = cv2.erode(face_mask, kernel, iterations=2)
        boundary_mask = cv2.subtract(face_mask, cv2.erode(face_mask, kernel, iterations=1))
        if cv2.countNonZero(inner_mask) < 250 or cv2.countNonZero(boundary_mask) < 80:
            return self._result("Insufficient face pixels for analysis")

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        inner_ab = np.array([cv2.mean(lab[:, :, c], mask=inner_mask)[0] for c in (1, 2)])
        boundary_ab = np.array([cv2.mean(lab[:, :, c], mask=boundary_mask)[0] for c in (1, 2)])
        chroma_delta = float(np.linalg.norm(inner_ab - boundary_ab))

        inner_texture = self._laplacian_variance(gray, inner_mask)
        boundary_texture = self._laplacian_variance(gray, boundary_mask)
        texture_delta = float(abs(inner_texture - boundary_texture) / (inner_texture + boundary_texture + 1e-6))

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)
        boundary_edge = float(cv2.mean(gradient, mask=boundary_mask)[0])
        inner_edge = float(cv2.mean(gradient, mask=inner_mask)[0])
        edge_ratio = float(boundary_edge / (inner_edge + 1e-6))

        chroma_signal = float(np.clip(chroma_delta / 22.0, 0.0, 1.0))
        texture_signal = float(np.clip(texture_delta / 0.70, 0.0, 1.0))
        edge_signal = float(np.clip(max(0.0, edge_ratio - 1.0) / 1.5, 0.0, 1.0))
        raw_score = 100.0 * (0.40 * chroma_signal + 0.35 * texture_signal + 0.25 * edge_signal)

        face_id = face_data.get("id") or face_data.get("face_id")
        if face_id is not None:
            previous = self.last_score.get(face_id, raw_score)
            a = self.smooth_factor
            smoothed = (1.0 - a) * previous + a * raw_score
            self.last_score[face_id] = smoothed
            signals_high = sum(v >= threshold for v, threshold in (
                (chroma_signal, 0.55), (texture_signal, 0.55), (edge_signal, 0.55)
            ))
            self.history[face_id].append({"score": float(smoothed), "signals_high": signals_high >= 2})
            records = list(self.history[face_id])
            persistent_count = sum(r["score"] >= 58.0 and r["signals_high"] for r in records)
            review = len(records) >= self.min_persistent_frames and persistent_count >= self.min_persistent_frames
            score = float(np.median([r["score"] for r in records]))
            stable_frames = len(records)
        else:
            score = float(raw_score)
            review = False
            stable_frames = 1

        return self._result(
            "Persistent multi-signal anomaly; review suggested" if review else
            "Experimental anomaly index only; not a real/fake probability",
            score=score,
            quality="usable",
            review_recommended=review,
            stable_frames=stable_frames,
            signals={
                "boundary_chroma_difference": round(chroma_delta, 2),
                "relative_texture_difference": round(texture_delta, 3),
                "boundary_edge_ratio": round(edge_ratio, 3),
                "chroma_signal_0_to_1": round(chroma_signal, 3),
                "texture_signal_0_to_1": round(texture_signal, 3),
                "edge_signal_0_to_1": round(edge_signal, 3),
                "blur_measure": round(blur, 1),
            },
        )

    def forget_face(self, face_id):
        """Remove temporal state when a tracker permanently drops a face ID."""
        self.history.pop(face_id, None)
        self.last_score.pop(face_id, None)
