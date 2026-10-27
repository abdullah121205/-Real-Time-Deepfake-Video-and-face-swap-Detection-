"""
Utility functions for face preprocessing, visual HUD rendering, and temporal smoothing.
"""

from collections import deque
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def crop_face_with_margin(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.25
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crops a face from an image with an additional percentage margin around borders.
    Face-swap and deepfake artifacts often appear at blending boundaries (hairline, chin, ears),
    so extra context is critical for accurate forensic analysis.

    Args:
        image: Source image as BGR numpy array.
        bbox: Bounding box tuple (x, y, w, h).
        margin: Fractional margin to add on each side (0.25 = 25% extra width/height).

    Returns:
        cropped_face: Cropped image region.
        expanded_bbox: The adjusted (x1, y1, x2, y2) coordinates clamped to image borders.
    """
    img_h, img_w = image.shape[:2]
    x, y, w, h = bbox

    dx = int(w * margin)
    dy = int(h * margin)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(img_w, x + w + dx)
    y2 = min(img_h, y + h + dy)

    # Fallback to prevent 0-sized crops
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = max(0, x), max(0, y), min(img_w, x + max(1, w)), min(img_h, y + max(1, h))

    cropped_face = image[y1:y2, x1:x2].copy()
    return cropped_face, (x1, y1, x2, y2)


def preprocess_face_for_model(
    face_bgr: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize_imagenet: bool = True
) -> np.ndarray:
    """
    Prepares a cropped BGR face image for deep learning model inference.
    Returns a normalized float32 tensor array in CHW format (shape: 3 x H x W).
    """
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Cannot preprocess an empty face image.")

    # Resize with high-quality Lanczos or cubic interpolation
    face_resized = cv2.resize(face_bgr, target_size, interpolation=cv2.INTER_AREA)

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if normalize_imagenet:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        face_rgb = (face_rgb - mean) / std

    # Transpose from HWC (H, W, C) to CHW (C, H, W)
    face_transposed = np.transpose(face_rgb, (2, 0, 1))
    return np.ascontiguousarray(face_transposed, dtype=np.float32)


class TemporalSmoother:
    """
    Eliminates frame-by-frame flicker in real-time video by smoothing confidence scores
    over a rolling temporal window using Exponential Moving Average (EMA) or moving average.
    """

    def __init__(self, window_size: int = 12, alpha: float = 0.35):
        """
        Args:
            window_size: Max number of history frames to retain.
            alpha: EMA weight for current frame (0.0 < alpha <= 1.0).
                   Higher values respond faster; lower values produce smoother outputs.
        """
        self.window_size = window_size
        self.alpha = alpha
        self.history: Dict[int, deque] = {}
        self.ema_values: Dict[int, float] = {}

    def update(self, face_id: int, raw_score: float) -> float:
        """
        Updates the history for a face ID and returns the temporally smoothed fake probability.

        Args:
            face_id: Unique face identifier in tracking.
            raw_score: Raw model output score [0.0 = Real, 1.0 = Fake].

        Returns:
            Smoothed fake probability score.
        """
        if face_id not in self.history:
            self.history[face_id] = deque(maxlen=self.window_size)
            self.ema_values[face_id] = raw_score

        self.history[face_id].append(raw_score)

        # Exponential moving average calculation
        current_ema = self.alpha * raw_score + (1.0 - self.alpha) * self.ema_values[face_id]
        self.ema_values[face_id] = current_ema

        return float(current_ema)

    def reset(self, face_id: Optional[int] = None):
        """Resets smoother state for a specific face or all faces."""
        if face_id is not None:
            self.history.pop(face_id, None)
            self.ema_values.pop(face_id, None)
        else:
            self.history.clear()
            self.ema_values.clear()


def draw_detection_hud(
    frame: np.ndarray,
    detections: List[Dict],
    fps: float = 0.0,
    latency_ms: float = 0.0,
    show_details: bool = True
) -> np.ndarray:
    """
    Renders an overlay HUD showing bounding boxes, status badges,
    confidence bars, and forensic metrics.

    Args:
        frame: Target image frame (BGR).
        detections: List of detection dicts containing:
                    - 'bbox': (x1, y1, x2, y2)
                    - 'fake_score': float in [0.0, 1.0]
                    - 'label': 'REAL' | 'DEEPFAKE' | 'FACE-SWAP' | 'UNCERTAIN'
                    - 'swap_score': float (optional)
                    - 'freq_score': float (optional)
        fps: Current frames-per-second.
        latency_ms: Total processing latency per frame in ms.
        show_details: Whether to render detailed forensic metrics.

    Returns:
        Frame with overlaid HUD elements.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Render each detected face
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        fake_score = det.get("fake_score", 0.0)
        label = det.get("label", "REAL" if fake_score < 0.5 else "DEEPFAKE")
        swap_score = det.get("swap_score", None)

        # Determine theme color based on classification
        # Colors are in BGR format
        if label in ("DEEPFAKE", "FACE-SWAP") or fake_score >= 0.65:
            primary_color = (40, 50, 230)      # Crimson / Red
            accent_color = (80, 80, 255)
            badge_text = f"MANIPULATED: {fake_score * 100:.1f}%"
        elif fake_score >= 0.40:
            primary_color = (0, 180, 255)       # Amber / Warning
            accent_color = (50, 210, 255)
            badge_text = f"SUSPICIOUS: {fake_score * 100:.1f}%"
        else:
            primary_color = (50, 205, 50)       # Emerald Green
            accent_color = (120, 240, 120)
            badge_text = f"AUTHENTIC: {(1.0 - fake_score) * 100:.1f}%"

        # 1. Sleek Corner Brackets
        corner_len = max(12, int(min(x2 - x1, y2 - y1) * 0.18))
        thickness = 2

        # Top-Left
        cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), primary_color, thickness)
        cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), primary_color, thickness)
        # Top-Right
        cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), primary_color, thickness)
        cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), primary_color, thickness)
        # Bottom-Left
        cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), primary_color, thickness)
        cv2.line(overlay, (x1, y2), (x1, y2 - corner_len), primary_color, thickness)
        # Bottom-Right
        cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), primary_color, thickness)
        cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), primary_color, thickness)

        # Faint bounding rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), primary_color, 1, cv2.LINE_AA)

        # 2. Status Badge Header
        badge_y = max(24, y1 - 10)
        (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        badge_bg_x2 = min(w - 2, x1 + text_w + 16)

        # Semi-transparent badge background
        cv2.rectangle(overlay, (x1, badge_y - text_h - 8), (badge_bg_x2, badge_y + 4), (20, 20, 20), -1)
        cv2.rectangle(overlay, (x1, badge_y - text_h - 8), (badge_bg_x2, badge_y + 4), primary_color, 1)

        cv2.putText(
            overlay,
            badge_text,
            (x1 + 8, badge_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            primary_color,
            2,
            cv2.LINE_AA
        )

        # 3. Confidence meter bar below bounding box
        bar_y = min(h - 10, y2 + 10)
        bar_w = max(60, x2 - x1)
        bar_h = 6

        # Meter background
        cv2.rectangle(overlay, (x1, bar_y), (x1 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Meter fill (length proportional to fake_score)
        fill_w = int(bar_w * np.clip(fake_score, 0.0, 1.0))
        cv2.rectangle(overlay, (x1, bar_y), (x1 + fill_w, bar_y + bar_h), primary_color, -1)

        # Optional detail telemetry under badge
        if show_details and swap_score is not None:
            detail_text = f"Swap Anomaly: {swap_score * 100:.0f}%"
            cv2.putText(
                overlay,
                detail_text,
                (x1, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                accent_color,
                1,
                cv2.LINE_AA
            )

    # 4. Top Telemetry Dashboard Card
    dash_w = 260
    dash_h = 80 if show_details else 50
    cv2.rectangle(overlay, (12, 12), (12 + dash_w, 12 + dash_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (12, 12), (12 + dash_w, 12 + dash_h), (60, 60, 60), 1)

    cv2.putText(
        overlay,
        "REAL-TIME DEEPFAKE DETECTOR",
        (22, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 215, 255),
        1,
        cv2.LINE_AA
    )

    status_line = f"FPS: {fps:4.1f} | Latency: {latency_ms:4.1f}ms"
    cv2.putText(
        overlay,
        status_line,
        (22, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    if show_details:
        face_count = len(detections)
        count_line = f"Active Faces: {face_count} | Mode: Live Analysis"
        cv2.putText(
            overlay,
            count_line,
            (22, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (160, 160, 160),
            1,
            cv2.LINE_AA
        )

    # Blend overlay with original frame for subtle transparency
    alpha = 0.92
    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    return blended
