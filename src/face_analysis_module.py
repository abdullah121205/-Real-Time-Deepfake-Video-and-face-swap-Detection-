
"""Integrated face analysis module for deepfake detection."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_detector import FaceDetector


class FaceAnalysisModule:
    """Integrates face detection, tracking, landmarks and artifact analysis."""

    def __init__(self, detector=None):
        if detector is not None:
            self.detector = detector
        else:
            old_cwd = Path.cwd()
            try:
                os.chdir(PROJECT_ROOT)
                self.detector = FaceDetector()
            finally:
                os.chdir(old_cwd)

        self.frame_index = 0

    @staticmethod
    def clean_analysis(analysis):
        if not isinstance(analysis, dict):
            return None

        signals = analysis.get("signals", {})
        if not isinstance(signals, dict):
            signals = {}

        return {
            "score": analysis.get("score"),
            "quality": analysis.get("quality", "insufficient"),
            "review_recommended": bool(
                analysis.get("review_recommended", False)
            ),
            "details": str(analysis.get("details", "")),
            "stable_frames": int(analysis.get("stable_frames", 0) or 0),
            "signals": signals,
            "classification": "not_assessed",
        }

    def process_frame(self, frame):
        """Analyze one frame and return its report and annotated image."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty OpenCV image")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have 3 color channels")

        faces = self.detector.detect_faces(frame)
        annotated = self.detector.annotate_frame(frame, faces)

        face_reports = []

        for face in faces:
            bbox = face.get("bbox", (0, 0, 0, 0))
            landmarks = face.get("landmarks") or []
            analysis = self.clean_analysis(face.get("swap_analysis"))

            face_reports.append({
                "face_id": face.get("id"),
                "bbox_xywh": [int(value) for value in bbox],
                "landmark_count": len(landmarks),
                "landmarks_xy": [
                    [int(point[0]), int(point[1])]
                    for point in landmarks
                ],
                "face_crop_available": (
                    isinstance(face.get("crop"), np.ndarray)
                    and face["crop"].size > 0
                ),
                "artifact_analysis": analysis,
            })

        usable_scores = [
            float(face["artifact_analysis"]["score"])
            for face in face_reports
            if face["artifact_analysis"] is not None
            and face["artifact_analysis"].get("quality") == "usable"
            and face["artifact_analysis"].get("score") is not None
        ]

        review_count = sum(
            1 for face in face_reports
            if face["artifact_analysis"] is not None
            and face["artifact_analysis"].get("review_recommended")
        )

        self.frame_index += 1

        report = {
            "frame_index": self.frame_index,
            "frame_size_wh": [int(frame.shape[1]), int(frame.shape[0])],
            "face_count": len(face_reports),
            "faces": face_reports,
            "summary": {
                "usable_artifact_analyses": len(usable_scores),
                "review_recommendations": review_count,
                "mean_artifact_index": (
                    round(float(np.mean(usable_scores)), 1)
                    if usable_scores else None
                ),
            },
            "limitations": (
                "Artifact indices are heuristic signals, not probabilities "
                "or verified real/fake labels. A review recommendation "
                "is not proof of manipulation."
            ),
        }

        return {
            "report": report,
            "annotated_frame": annotated,
        }

    def process_video(
        self,
        source=0,
        display=True,
        output_json=None,
        max_frames=None,
    ):
        """Process a webcam or video and optionally save JSON results."""
        capture = cv2.VideoCapture(source)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video source: {source}")

        reports = []
        window_name = "Face Analysis Module - press Q to quit"

        try:
            if display:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while max_frames is None or len(reports) < max_frames:
                # Stop if the user has closed the display window.
                if display:
                    try:
                        if cv2.getWindowProperty(
                            window_name, cv2.WND_PROP_VISIBLE
                        ) < 1:
                            break
                    except cv2.error:
                        break

                ok, frame = capture.read()
                if not ok:
                    break

                # Flip only the webcam preview source, not video files.
                # This makes the webcam image unmirrored.
                if isinstance(source, int) and source == 0:
                    frame = cv2.flip(frame, 1)

                result = self.process_frame(frame)
                reports.append(result["report"])

                if display:
                    cv2.imshow(window_name, result["annotated_frame"])
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q") or key == 27:
                        break

                    # Detect clicking the window's X button.
                    try:
                        if cv2.getWindowProperty(
                            window_name, cv2.WND_PROP_VISIBLE
                        ) < 1:
                            break
                    except cv2.error:
                        break

        finally:
            capture.release()
            if display:
                cv2.destroyAllWindows()
                cv2.waitKey(1)

        usable_scores = [
            face["artifact_analysis"]["score"]
            for report in reports
            for face in report["faces"]
            if face["artifact_analysis"] is not None
            and face["artifact_analysis"].get("quality") == "usable"
            and face["artifact_analysis"].get("score") is not None
        ]

        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "frames_processed": len(reports),
            "faces_seen_across_frames": sum(
                report["face_count"] for report in reports
            ),
            "frames_with_review_recommendations": sum(
                1 for report in reports
                if report["summary"]["review_recommendations"] > 0
            ),
            "mean_usable_artifact_index": (
                round(float(np.mean(usable_scores)), 1)
                if usable_scores else None
            ),
            "score_interpretation": (
                "Experimental heuristic index; not a probability "
                "or validated classifier output."
            ),
            "frames": reports,
        }

        if output_json:
            destination = Path(output_json)
            if not destination.is_absolute():
                destination = Path.cwd() / destination

            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            summary["saved_json"] = str(destination)

        return summary


def parse_source(value):
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(
        description="Run the integrated face-analysis module"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or video file path",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for a JSON analysis report",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening a display window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional maximum number of frames",
    )

    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")

    module = FaceAnalysisModule()

    summary = module.process_video(
        source=parse_source(args.source),
        display=not args.no_display,
        output_json=args.output_json,
        max_frames=args.max_frames,
    )

    print(json.dumps({
        key: value
        for key, value in summary.items()
        if key != "frames"
    }, indent=2))

    if args.output_json:
        print("Full report saved to:", summary.get(
            "saved_json", args.output_json
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())