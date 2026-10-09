
import cv2
from pathlib import Path


class FaceDetector:
    """Detect and extract faces from images and video frames using OpenCV."""

    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(40, 40)):
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("Could not load OpenCV face detector.")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_faces(self, frame):
        """Return bounding boxes and cropped face images."""
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
            face_crop = frame[y:y + height, x:x + width].copy()

            faces.append({
                "bbox": (int(x), int(y), int(width), int(height)),
                "crop": face_crop
            })

        return faces

    def annotate_frame(self, frame, faces):
        """Draw face boxes and return an annotated copy."""
        output = frame.copy()

        for index, face in enumerate(faces, start=1):
            x, y, width, height = face["bbox"]

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"Face {index}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return output


def process_video(source=0):
    """
    Process a webcam (source=0) or a video file path.
    Press Q to exit.
    """
    detector = FaceDetector()
    video = cv2.VideoCapture(source)

    if not video.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    try:
        while True:
            success, frame = video.read()

            if not success:
                break

            faces = detector.detect_faces(frame)
            annotated = detector.annotate_frame(frame, faces)

            cv2.putText(
                annotated,
                f"Faces detected: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("OpenCV Face Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect faces in a video or webcam stream."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0) or path to a video file."
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    process_video(source)
