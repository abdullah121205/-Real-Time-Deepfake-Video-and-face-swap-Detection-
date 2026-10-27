import cv2
import math
import numpy as np
from face_swap_detector import FaceSwapDetector

class FaceTracker:
    def __init__(self, max_distance=150, max_disappeared=10, smooth_factor=0.3):
        self.next_id = 1
        self.tracked_faces = {}  # Maps ID to (center_x, center_y, width, height)
        self.disappeared = {}    # Maps ID to number of lost frames
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.smooth_factor = smooth_factor

    def update(self, faces):
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
            min_dist = float('inf')

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
                # Smooth the movement using Exponential Moving Average to prevent jitter
                px, py, pw, ph = self.tracked_faces[best_id]
                center_x = int(px + self.smooth_factor * (center_x - px))
                center_y = int(py + self.smooth_factor * (center_y - py))
                width = int(pw + self.smooth_factor * (width - pw))
                height = int(ph + self.smooth_factor * (height - ph))
                
                # Update the face bbox to the smoothed version
                face["bbox"] = (center_x - width // 2, center_y - height // 2, width, height)

            updated_tracked_faces[best_id] = (center_x, center_y, width, height)
            self.disappeared[best_id] = 0
            matched_ids.add(best_id)
            face["id"] = best_id

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
    """Detect and extract faces from video frames using OpenCV."""

    def __init__(self, scale_factor=1.1, min_neighbors=8,
                 min_size=(100, 100)):
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("Failed to load OpenCV face detector.")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.tracker = FaceTracker()
        
        # Initialize OpenCV Facemark for landmark detection
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel("src/lbfmodel.yaml")
        
        # Initialize Face Swap Detector
        self.swap_detector = FaceSwapDetector()

    def detect_faces(self, frame):
        """Return bounding boxes and cropped images for detected faces."""
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        faces = []

        for x, y, width, height in boxes:
            faces.append({
                "bbox": (int(x), int(y), int(width), int(height)),
                "crop": frame[y:y + height, x:x + width].copy()
            })

        # Track faces to maintain consistent IDs
        faces = self.tracker.update(faces)

        # Extract facial landmarks using OpenCV Facemark
        if len(faces) > 0:
            boxes = np.array([face["bbox"] for face in faces])
            success, landmarks = self.facemark.fit(gray, boxes)
            
            if success:
                for i, face in enumerate(faces):
                    # OpenCV sometimes nests the array weirdly like (68, 1, 2). Squeeze flattens it to (68, 2)
                    pts = np.squeeze(landmarks[i]).astype(int)
                    
                    # Ensure it has exactly 2 dimensions (N, 2) before saving
                    if pts.ndim == 2 and pts.shape[1] == 2:
                        face["landmarks"] = [(pt[0], pt[1]) for pt in pts]
                        # Analyze for face swap artifacts
                        face["swap_analysis"] = self.swap_detector.analyze_face(frame, face)
                    else:
                        face["landmarks"] = []
                        face["swap_analysis"] = None
            else:
                for face in faces:
                    face["landmarks"] = []
                    face["swap_analysis"] = None

        return faces

    def annotate_frame(self, frame, faces):
        """Draw bounding boxes around detected faces."""
        output = frame.copy()

        for face in faces:
            x, y, width, height = face["bbox"]
            face_id = face.get("id", "?")

            # Show an experimental artifact score, not a real/fake verdict.
            # The current handcrafted signals are not trained or validated enough
            # to reliably classify a face as genuine or manipulated.
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
            color = (0, 200, 255)  # Neutral amber/yellow; never labels a face real/fake

            thickness = 2
            
            # Length of the corner brackets (20% of the box width)
            length = int(width * 0.2)

            # Draw Top-left corner
            cv2.line(output, (x, y), (x + length, y), color, thickness)
            cv2.line(output, (x, y), (x, y + length), color, thickness)

            # Draw Top-right corner
            cv2.line(output, (x + width, y), (x + width - length, y), color, thickness)
            cv2.line(output, (x + width, y), (x + width, y + length), color, thickness)

            # Draw Bottom-left corner
            cv2.line(output, (x, y + height), (x + length, y + height), color, thickness)
            cv2.line(output, (x, y + height), (x, y + height - length), color, thickness)

            # Draw Bottom-right corner
            cv2.line(output, (x + width, y + height), (x + width - length, y + height), color, thickness)
            cv2.line(output, (x + width, y + height), (x + width, y + height - length), color, thickness)

            # Professional Label Background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 2)
            
            # Draw solid background rectangle for the text
            cv2.rectangle(output, (x, max(y - text_h - 10, 0)), (x + text_w + 10, max(y, text_h + 10)), color, -1)

            # Draw text over the background
            text_color = (0, 0, 0)
            cv2.putText(
                output,
                text,
                (x + 5, max(y - 5, text_h + 5)),
                font,
                font_scale,
                text_color, 
                2
            )

            # Landmarks remain available in face["landmarks"] for analysis,
            # but are intentionally not drawn to keep the webcam view clean.

        return output


def process_video(source=0):
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

            # Flip the frame horizontally for a mirrored view
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

            # Close if 'q' is pressed on the keyboard
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
            # Close if the 'X' button on the window is clicked
            if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or path to a video file"
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    process_video(source)