"""
Helper script to generate synthetic test media (images and video)
for testing the real-time deepfake and face-swap detection system.
"""

from pathlib import Path
import cv2
import numpy as np


def generate_face_frame(
    width: int = 1280,
    height: int = 720,
    face_pos: tuple = (640, 360),
    is_manipulated: bool = False,
    frame_idx: int = 0
) -> np.ndarray:
    """Creates a synthetic frame with realistic facial geometry and optional face-swap artifacts."""
    frame = np.full((height, width, 3), (35, 35, 40), dtype=np.uint8)

    # Slight background gradient
    for y in range(height):
        factor = 0.8 + 0.4 * (y / height)
        frame[y, :] = np.clip(frame[y, :] * factor, 0, 255).astype(np.uint8)

    fx, fy = face_pos
    # Slight breathing movement
    fy += int(4 * np.sin(frame_idx * 0.15))

    # Head / Neck base
    cv2.rectangle(frame, (fx - 45, fy + 80), (fx + 45, fy + 180), (140, 160, 190), -1)

    # Face Oval
    face_color = (165, 185, 215)
    cv2.ellipse(frame, (fx, fy), (90, 120), 0, 0, 360, face_color, -1)

    # Hair
    cv2.ellipse(frame, (fx, fy - 60), (95, 75), 0, 180, 360, (30, 25, 20), -1)

    # Eyes
    eye_offset = 35
    eye_y = fy - 20
    cv2.circle(frame, (fx - eye_offset, eye_y), 14, (240, 240, 240), -1)
    cv2.circle(frame, (fx + eye_offset, eye_y), 14, (240, 240, 240), -1)

    # Pupils with micro-saccade motion
    pupil_dx = int(2 * np.cos(frame_idx * 0.1))
    cv2.circle(frame, (fx - eye_offset + pupil_dx, eye_y), 6, (40, 30, 20), -1)
    cv2.circle(frame, (fx + eye_offset + pupil_dx, eye_y), 6, (40, 30, 20), -1)

    # Eyebrows
    cv2.line(frame, (fx - 55, eye_y - 18), (fx - 15, eye_y - 20), (30, 25, 20), 4)
    cv2.line(frame, (fx + 15, eye_y - 20), (fx + 55, eye_y - 18), (30, 25, 20), 4)

    # Nose
    cv2.line(frame, (fx, fy - 10), (fx, fy + 20), (130, 150, 180), 3)
    cv2.line(frame, (fx - 10, fy + 20), (fx + 10, fy + 20), (130, 150, 180), 3)

    # Mouth
    mouth_y = fy + 55
    cv2.ellipse(frame, (fx, mouth_y), (35, 12), 0, 0, 180, (60, 60, 160), -1)

    # Introduce synthetic face-swap artifacts if requested
    if is_manipulated:
        # 1. Blending boundary seam / halo ring around face core
        cv2.ellipse(frame, (fx, fy), (82, 110), 0, 0, 360, (190, 140, 120), 4)
        # 2. Chromatic disparity inside facial mask
        cv2.circle(frame, (fx, fy + 10), 55, (130, 140, 230), 2)
        # 3. Frequency / blur inconsistency
        blur_roi = frame[fy - 50:fy + 70, fx - 60:fx + 60]
        blurred = cv2.GaussianBlur(blur_roi, (15, 15), 5.0)
        frame[fy - 50:fy + 70, fx - 60:fx + 60] = cv2.addWeighted(blur_roi, 0.35, blurred, 0.65, 0)

    return frame


def create_demo_assets(output_dir: str = "data"):
    """Creates a sample demo video (data/demo_sample.mp4) and images."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Create a 5-second 25fps demo video (125 frames)
    video_file = out_path / "demo_sample.mp4"
    width, height = 1280, 720
    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    print(f"[Demo] Generating 5-second test video: {video_file}...")
    num_frames = 125  # 5 seconds at 25 fps

    for i in range(num_frames):
        # First 60 frames authentic, then 65 frames with face-swap artifacts
        is_manipulated = (i >= 60)
        frame = generate_face_frame(width, height, is_manipulated=is_manipulated, frame_idx=i)
        writer.write(frame)

    writer.release()
    print(f"[Demo] Demo video saved successfully ({num_frames} frames).")

    # 2. Save sample real and fake static face frames
    real_sample_file = out_path / "sample_real.jpg"
    fake_sample_file = out_path / "sample_fake.jpg"

    real_frame = generate_face_frame(width, height, is_manipulated=False, frame_idx=10)
    fake_frame = generate_face_frame(width, height, is_manipulated=True, frame_idx=80)

    cv2.imwrite(str(real_sample_file), real_frame)
    cv2.imwrite(str(fake_sample_file), fake_frame)
    print(f"[Demo] Saved sample images: {real_sample_file}, {fake_sample_file}")


if __name__ == "__main__":
    create_demo_assets()
