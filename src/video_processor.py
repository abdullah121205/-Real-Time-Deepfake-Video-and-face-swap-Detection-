"""
Real-time video processing engine for deepfake and face-swap detection.
Coordinates face detection, forensic classifiers, temporal smoothing, and HUD rendering.
"""

from pathlib import Path
import time
from typing import Dict, Generator, List, Optional, Tuple, Union
import cv2
import numpy as np

from .deepfake_detector import DeepfakeDetector
from .face_detector import FaceDetector
from .face_swap_detector import FaceSwapDetector
from .utils import TemporalSmoother, draw_detection_hud


class VideoProcessor:
    """
    Core video analysis pipeline that processes live camera streams or pre-recorded videos.
    """

    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        deepfake_detector: Optional[DeepfakeDetector] = None,
        face_swap_detector: Optional[FaceSwapDetector] = None,
        temporal_smoother: Optional[TemporalSmoother] = None,
        frame_stride: int = 1,
        deepfake_weight: float = 0.65,
        swap_weight: float = 0.35
    ):
        """
        Args:
            face_detector: Face detector instance (defaults to Haar cascade).
            deepfake_detector: Deepfake classifier instance (defaults to Meso4).
            face_swap_detector: Face swap forensic analyzer.
            temporal_smoother: Smoother to eliminate score flickering across frames.
            frame_stride: Run deep inference every Nth frame (1 = every frame, 2 = alternate frames).
            deepfake_weight: Weight given to deep learning classifier score.
            swap_weight: Weight given to forensic face-swap anomaly score.
        """
        self.face_detector = face_detector or FaceDetector(backend="haar")
        self.deepfake_detector = deepfake_detector or DeepfakeDetector(model_name="meso4")
        self.face_swap_detector = face_swap_detector or FaceSwapDetector()
        self.smoother = temporal_smoother or TemporalSmoother(window_size=12, alpha=0.35)

        self.frame_stride = max(1, frame_stride)
        self.w_deepfake = deepfake_weight
        self.w_swap = swap_weight

        # Caching for skipped frames
        self._cached_detections: List[Dict] = []
        self._frame_count = 0

    def process_frame(self, frame_bgr: np.ndarray, show_details: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Processes a single video frame and returns the annotated frame with telemetry.

        Args:
            frame_bgr: Raw input BGR frame from camera or video.
            show_details: Whether to render detailed forensic metrics.

        Returns:
            annotated_frame: Frame with HUD overlay.
            telemetry: Dictionary containing processing statistics and detections.
        """
        t_start = time.perf_counter()
        self._frame_count += 1
        is_inference_frame = (self._frame_count % self.frame_stride == 0)

        current_detections: List[Dict] = []

        if is_inference_frame:
            # 1. Detect faces
            faces = self.face_detector.detect_faces(frame_bgr)

            for face in faces:
                crop = face["face_crop"]
                face_id = face["face_id"]

                # 2. Deep learning classification
                df_score, _ = self.deepfake_detector.predict(crop)

                # 3. Forensic face-swap analysis
                swap_report = self.face_swap_detector.analyze_face(crop)
                swap_score = swap_report["swap_score"]

                # 4. Composite manipulation score
                combined_score = (self.w_deepfake * df_score) + (self.w_swap * swap_score)

                # 5. Temporal smoothing
                smoothed_score = self.smoother.update(face_id, combined_score)

                # 6. Assign final label
                if smoothed_score >= 0.55:
                    label = "FACE-SWAP" if swap_score > 0.60 else "DEEPFAKE"
                elif smoothed_score >= 0.40:
                    label = "SUSPICIOUS"
                else:
                    label = "REAL"

                current_detections.append({
                    "face_id": face_id,
                    "bbox": face["bbox"],
                    "raw_bbox": face["raw_bbox"],
                    "fake_score": smoothed_score,
                    "raw_score": combined_score,
                    "deepfake_score": df_score,
                    "swap_score": swap_score,
                    "boundary_anomaly": swap_report["boundary_anomaly"],
                    "freq_score": swap_report["frequency_anomaly"],
                    "label": label
                })

            self._cached_detections = current_detections
        else:
            # On stride-skipped frames, reuse cached detections
            current_detections = self._cached_detections

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / max(1.0, latency_ms)

        # Render HUD elements
        annotated_frame = draw_detection_hud(
            frame=frame_bgr,
            detections=current_detections,
            fps=fps,
            latency_ms=latency_ms,
            show_details=show_details
        )

        telemetry = {
            "fps": fps,
            "latency_ms": latency_ms,
            "num_faces": len(current_detections),
            "detections": current_detections,
            "frame_index": self._frame_count
        }

        return annotated_frame, telemetry

    def stream_video(
        self,
        source: Union[int, str] = 0,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        Generator that reads from a video source and yields processed frames.

        Args:
            source: 0 for webcam or path string to video file.
            output_path: Path to write the output video file (optional).
            max_frames: Max number of frames to process before stopping.

        Yields:
            Tuple of (annotated_frame, telemetry)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_processed = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames_processed += 1
                annotated_frame, telemetry = self.process_frame(frame)

                if writer:
                    writer.write(annotated_frame)

                yield annotated_frame, telemetry

                if max_frames and frames_processed >= max_frames:
                    break
        finally:
            cap.release()
            if writer:
                writer.release()
                print(f"[VideoProcessor] Saved output video to: {output_path}")

    def run_window(
        self,
        source: Union[int, str] = 0,
        window_title: str = "Real-Time Deepfake & Face-Swap Detector",
        output_path: Optional[str] = None
    ):
        """
        Runs an interactive OpenCV GUI display window.
        Keys:
            'q': Quit
            's': Save current frame screenshot to results/
            'd': Toggle forensic details panel
        """
        print(f"[VideoProcessor] Starting live feed from source '{source}'...")
        print("[VideoProcessor] Controls: [Q] Quit | [S] Save Screenshot | [D] Toggle Details")

        show_details = True
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, telemetry = self.process_frame(frame, show_details=show_details)

                if writer:
                    writer.write(annotated)

                cv2.imshow(window_title, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    shot_path = Path("results") / f"screenshot_{int(time.time())}.jpg"
                    shot_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(shot_path), annotated)
                    print(f"[VideoProcessor] Snapshot saved to: {shot_path}")
                elif key == ord("d"):
                    show_details = not show_details
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("[VideoProcessor] Finished session.")
