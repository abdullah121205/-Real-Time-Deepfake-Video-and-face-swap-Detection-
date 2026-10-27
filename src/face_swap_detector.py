"""Conservative, interpretable face-swap artifact screening.

This is a research heuristic, not a trained deepfake classifier. Its score is
an anomaly index (not a probability). A review signal is raised only when
multiple weak signals persist across several frames. Validate on labelled,
representative real and manipulated videos before making accuracy claims.
"""

from collections import defaultdict, deque

import cv2
import numpy as np


class FaceSwapDetector:
    """Estimate persistent visual anomalies in landmark-defined face regions."""

    def __init__(self, history_length=12, min_persistent_frames=6, smooth_factor=0.25):
        self.history_length = max(4, int(history_length))
        self.min_persistent_frames = max(2, min(int(min_persistent_frames), self.history_length))
        self.smooth_factor = float(np.clip(smooth_factor, 0.0, 1.0))
        self.history = defaultdict(lambda: deque(maxlen=self.history_length))
        self.last_score = {}

    @staticmethod
    def _result(details, score=0.0, quality="insufficient", signals=None,
                review_recommended=False, stable_frames=0):
        return {
            "score": round(float(score), 1),
            "is_fake": None,
            "review_recommended": bool(review_recommended),
            "quality": quality,
            "stable_frames": int(stable_frames),
            "details": details,
            "signals": signals or {},
        }

    @staticmethod
    def _masked_mean(image, mask):
        values = cv2.mean(image, mask=mask)
        return np.asarray(values[:image.shape[2] if image.ndim == 3 else 1], dtype=np.float32)

    @staticmethod
    def _laplacian_variance(gray, mask):
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        values = lap[mask > 0]
        if values.size < 25:
            return 0.0
        return float(np.var(values))

    @staticmethod
    def _mask_mean(image, mask):
        return float(cv2.mean(image, mask=mask)[0])

    def analyze_face(self, frame, face_data):
        """Return anomaly signals and a conservative multi-frame review hint."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return self._result("Invalid frame")

        landmarks = face_data.get("landmarks", [])
        if landmarks is None or len(landmarks) < 68:
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

        # Reject implausible landmark sets rather than clipping bad coordinates
        # into an artificial polygon that could produce misleading scores.
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

        # Quality checks. Low quality should suppress a score, not count as fake.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        if blur < 18:
            return self._result("Low-detail or blurred face; assessment skipped", quality="low")
        if brightness < 25 or brightness > 232:
            return self._result("Very dark or overexposed face; assessment skipped", quality="low")

        # Inner face area and narrow contour ring. This ring is only a proxy for
        # boundaries; hair, pose and occlusion can affect it, so it is weak evidence.
        k = max(3, (min(roi.shape[:2]) // 20) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        inner_mask = cv2.erode(face_mask, kernel, iterations=2)
        boundary_mask = cv2.subtract(face_mask, cv2.erode(face_mask, kernel, iterations=1))
        if cv2.countNonZero(inner_mask) < 250 or cv2.countNonZero(boundary_mask) < 80:
            return self._result("Insufficient face pixels for analysis")

        # LAB separates chroma (a,b) from brightness (L), reducing but not
        # eliminating illumination sensitivity.
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        inner_ab = np.array([cv2.mean(lab[:, :, c], mask=inner_mask)[0] for c in (1, 2)])
        boundary_ab = np.array([cv2.mean(lab[:, :, c], mask=boundary_mask)[0] for c in (1, 2)])
        chroma_delta = float(np.linalg.norm(inner_ab - boundary_ab))

        inner_texture = self._laplacian_variance(gray, inner_mask)
        boundary_texture = self._laplacian_variance(gray, boundary_mask)
        texture_delta = float(abs(inner_texture - boundary_texture) / (inner_texture + boundary_texture + 1e-6))

        # Edge concentration in the contour ring compared with the inner region.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)
        boundary_edge = float(cv2.mean(gradient, mask=boundary_mask)[0])
        inner_edge = float(cv2.mean(gradient, mask=inner_mask)[0])
        edge_ratio = float(boundary_edge / (inner_edge + 1e-6))

        # Conservative, interpretable feature scaling. These constants are
        # starting heuristics and must be calibrated on a labelled dataset.
        chroma_signal = float(np.clip(chroma_delta / 22.0, 0.0, 1.0))
        texture_signal = float(np.clip(texture_delta / 0.70, 0.0, 1.0))
        edge_signal = float(np.clip(max(0.0, edge_ratio - 1.0) / 1.5, 0.0, 1.0))
        raw_score = 100.0 * (0.40 * chroma_signal + 0.35 * texture_signal + 0.25 * edge_signal)

        # A stable face ID allows temporal aggregation. Without an ID, we report
        # only a single-frame index and never recommend review from one frame.
        face_id = face_data.get("id")
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
