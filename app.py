"""
Real-Time Deepfake Video and Face-Swap Detection Application.
Main entry point for live webcam analysis, video verification, and benchmark tests.
"""

import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

from src.deepfake_detector import DeepfakeDetector
from src.face_detector import FaceDetector
from src.face_swap_detector import FaceSwapDetector
from src.utils import TemporalSmoother
from src.video_processor import VideoProcessor


def run_benchmark(model_name: str = "meso4", num_frames: int = 100):
    """
    Runs an automated performance and latency benchmark
    simulating live video frames at 720p (1280x720) with multiple faces.
    """
    print("=" * 60)
    print(f"  RUNNING SYSTEM BENCHMARK: {model_name.upper()}")
    print("=" * 60)

    # Initialize components
    face_detector = FaceDetector(backend="haar")
    deepfake_detector = DeepfakeDetector(model_name=model_name)
    face_swap_detector = FaceSwapDetector()
    smoother = TemporalSmoother()

    processor = VideoProcessor(
        face_detector=face_detector,
        deepfake_detector=deepfake_detector,
        face_swap_detector=face_swap_detector,
        temporal_smoother=smoother,
        frame_stride=1
    )

    # Generate synthetic 720p test frames with simulated faces
    print(f"[Benchmark] Processing {num_frames} synthetic frames (1280x720)...")
    simulated_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    # Draw synthetic face-like shapes
    cv2.circle(simulated_frame, (640, 360), 120, (180, 200, 230), -1)
    cv2.circle(simulated_frame, (590, 320), 15, (50, 50, 50), -1)
    cv2.circle(simulated_frame, (690, 320), 15, (50, 50, 50), -1)
    cv2.ellipse(simulated_frame, (640, 420), (50, 20), 0, 0, 180, (40, 40, 150), -1)

    latencies = []
    t_start = time.perf_counter()

    for i in range(num_frames):
        t0 = time.perf_counter()
        processor.process_frame(simulated_frame, show_details=True)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    total_time = time.perf_counter() - t_start
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    effective_fps = num_frames / total_time

    print("\n" + "-" * 40)
    print("  BENCHMARK RESULTS")
    print("-" * 40)
    print(f"Device:            {deepfake_detector.device}")
    print(f"Total Frames:      {num_frames}")
    print(f"Total Time:        {total_time:.2f} s")
    print(f"Average Latency:   {avg_latency:.2f} ms / frame")
    print(f"95th Pct Latency:  {p95_latency:.2f} ms / frame")
    print(f"Effective Throughput: {effective_fps:.1f} FPS")
    print("-" * 40)
    if effective_fps >= 25.0:
        print("Status: EXCELLENT. Ready for real-time video streaming (>= 25 FPS).")
    else:
        print("Status: GOOD. Recommend using --stride 2 for high frame-rate cameras.")
    print("=" * 60)


def export_onnx(model_name: str, output_path: str):
    """Exports PyTorch deepfake model to ONNX format for accelerated deployment."""
    print(f"[ONNX Export] Initializing {model_name}...")
    try:
        detector = DeepfakeDetector(model_name=model_name, device="cpu")
        dummy_input = torch.randn(1, 3, 224, 224)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            detector.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"[ONNX Export] Successfully exported model to: {output_path}")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[ONNX Export] Error exporting ONNX: {e}")
        print("Note: To export models to ONNX with PyTorch, please install: pip install onnx onnxscript")
    except Exception as e:
        print(f"[ONNX Export] Failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Deepfake Video and Face-Swap Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video source: webcam index (e.g. '0') or path to video file (e.g. 'data/test.mp4')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meso4",
        choices=["meso4", "meso_inception", "mobilenet"],
        help="Deepfake neural network model architecture"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained model weights (.pth). If omitted, model runs with forensic calibration."
    )
    parser.add_argument(
        "--face-detector",
        type=str,
        default="haar",
        choices=["haar", "yunet"],
        help="Face detector backend: 'haar' (fast CPU) or 'yunet' (high-accuracy DNN)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run inference every Nth frame (1 = every frame, 2 = alternate frames for higher FPS)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save annotated output video (e.g. 'results/output.mp4')"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run an automated synthetic performance benchmark without opening a camera"
    )
    parser.add_argument(
        "--export-onnx",
        type=str,
        default=None,
        help="Export the model to specified ONNX path and exit (e.g. 'models/detector.onnx')"
    )

    args = parser.parse_args()

    # Mode 1: Benchmark
    if args.benchmark:
        run_benchmark(model_name=args.model)
        return

    # Mode 2: Export ONNX
    if args.export_onnx:
        export_onnx(model_name=args.model, output_path=args.export_onnx)
        return

    # Mode 3: Live Video / Camera Analysis
    print("=" * 60)
    print("  REAL-TIME DEEPFAKE & FACE-SWAP DETECTION SYSTEM")
    print("=" * 60)
    print(f"Source:        {args.source}")
    print(f"Model:         {args.model.upper()}")
    print(f"Face Backend:  {args.face_detector.upper()}")
    print(f"Frame Stride:  {args.stride}")
    if args.weights:
        print(f"Weights:       {args.weights}")
    if args.save:
        print(f"Output Video:  {args.save}")
    print("=" * 60)

    # Resolve source (integer index for webcam or string path for video file)
    src: Union[int, str]
    if args.source.isdigit():
        src = int(args.source)
    else:
        src = args.source
        if not Path(src).is_file():
            print(f"[Error] Specified video file does not exist: {src}")
            sys.exit(1)

    face_detector = FaceDetector(backend=args.face_detector)
    deepfake_detector = DeepfakeDetector(model_name=args.model, weights_path=args.weights)
    face_swap_detector = FaceSwapDetector()
    smoother = TemporalSmoother(window_size=12, alpha=0.35)

    processor = VideoProcessor(
        face_detector=face_detector,
        deepfake_detector=deepfake_detector,
        face_swap_detector=face_swap_detector,
        temporal_smoother=smoother,
        frame_stride=args.stride
    )

    try:
        processor.run_window(source=src, output_path=args.save)
    except Exception as e:
        print(f"\n[System Error] {e}")
        print("\nTip: If you do not have an active webcam connected, test the pipeline with:")
        print("  python app.py --benchmark")
        print("or specify a video file path:")
        print("  python app.py --source path/to/video.mp4")


if __name__ == "__main__":
    main()
