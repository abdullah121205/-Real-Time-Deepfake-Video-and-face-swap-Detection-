"""
Comprehensive unit and integration tests for the Deepfake & Face-Swap Detection system.
"""

import unittest
import numpy as np
import torch

from src.deepfake_detector import DeepfakeDetector, Meso4, MesoInception4, MobileNetDeepfake
from src.face_detector import FaceDetector
from src.face_swap_detector import FaceSwapDetector
from src.utils import (
    crop_face_with_margin,
    draw_detection_hud,
    preprocess_face_for_model,
    TemporalSmoother,
)
from src.video_processor import VideoProcessor


class TestDeepfakeDetectionSystem(unittest.TestCase):
    """Test suite for core components, neural networks, and video pipelines."""

    def setUp(self):
        # Create a synthetic 480x640 BGR image
        self.test_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Create a synthetic 100x100 BGR face crop
        self.test_face = np.full((100, 100, 3), 160, dtype=np.uint8)

    def test_crop_face_with_margin(self):
        """Verify face cropping correctly expands bounding box and clamps to limits."""
        bbox = (100, 100, 80, 80)
        cropped, expanded = crop_face_with_margin(self.test_frame, bbox, margin=0.25)

        self.assertIsNotNone(cropped)
        self.assertEqual(len(expanded), 4)
        x1, y1, x2, y2 = expanded
        # Expanded box should be wider and taller than original
        self.assertLess(x1, 100)
        self.assertLess(y1, 100)
        self.assertGreater(x2, 180)
        self.assertGreater(y2, 180)

    def test_preprocess_face_for_model(self):
        """Verify image preprocessing output shape and normalization."""
        tensor = preprocess_face_for_model(self.test_face, target_size=(224, 224))
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

    def test_temporal_smoother(self):
        """Verify EMA smoothing dampens rapid score fluctuations."""
        smoother = TemporalSmoother(window_size=10, alpha=0.3)
        # Alternate between 0.1 and 0.9 to simulate jitter
        s1 = smoother.update(face_id=0, raw_score=0.1)
        s2 = smoother.update(face_id=0, raw_score=0.9)
        s3 = smoother.update(face_id=0, raw_score=0.1)

        # Smoothed score should be dampening extremes
        self.assertLess(s2, 0.9)
        self.assertGreater(s3, 0.1)

    def test_meso4_forward_pass(self):
        """Verify Meso4 architecture output dimension."""
        model = Meso4(num_classes=1)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 1))

    def test_meso_inception4_forward_pass(self):
        """Verify MesoInception4 architecture output dimension."""
        model = MesoInception4(num_classes=1)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 1))

    def test_mobilenet_forward_pass(self):
        """Verify MobileNetDeepfake architecture output dimension."""
        model = MobileNetDeepfake(num_classes=1, pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 1))

    def test_deepfake_detector_prediction(self):
        """Verify DeepfakeDetector inference outputs bounded probabilities."""
        detector = DeepfakeDetector(model_name="meso4", device="cpu")
        score, label = detector.predict(self.test_face)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIn(label, ["AUTHENTIC", "DEEPFAKE"])

    def test_face_swap_detector(self):
        """Verify FaceSwapDetector produces complete forensic report."""
        detector = FaceSwapDetector()
        report = detector.analyze_face(self.test_face)

        self.assertIn("swap_score", report)
        self.assertIn("score", report)
        self.assertIn("boundary_anomaly", report)
        self.assertIn("frequency_anomaly", report)
        self.assertIn("color_disparity", report)

        for metric in ["swap_score", "boundary_anomaly", "frequency_anomaly", "color_disparity"]:
            self.assertGreaterEqual(report[metric], 0.0)
            self.assertLessEqual(report[metric], 1.0)

        self.assertGreaterEqual(report["score"], 0.0)
        self.assertLessEqual(report["score"], 100.0)

    def test_hud_rendering(self):
        """Verify HUD rendering returns valid uint8 image with identical shape."""
        sample_dets = [{
            "bbox": (100, 100, 250, 250),
            "fake_score": 0.88,
            "label": "DEEPFAKE",
            "swap_score": 0.75
        }]
        annotated = draw_detection_hud(self.test_frame, sample_dets, fps=30.0, latency_ms=15.0)
        self.assertEqual(annotated.shape, self.test_frame.shape)
        self.assertEqual(annotated.dtype, np.uint8)

    def test_video_processor_end_to_end(self):
        """Verify full end-to-end processing pipeline on a single frame."""
        processor = VideoProcessor()
        annotated, telemetry = processor.process_frame(self.test_frame)

        self.assertEqual(annotated.shape, self.test_frame.shape)
        self.assertIn("fps", telemetry)
        self.assertIn("latency_ms", telemetry)
        self.assertIn("detections", telemetry)


if __name__ == "__main__":
    unittest.main()
